# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Pre-screen candidate precision/engineering/boundaries with frozen artifacts."""

import argparse
from pathlib import Path
import sys
from typing import Any

from evidence import _candidate, _child, _environment, _read, _write
from run_matrix import _freeze_files, _snapshot

HERE = Path(__file__).resolve().parent


def _supplied_plan(entry: dict[str, Any]) -> Path:
    name = entry.get("supplied_outputs_plan")
    if name:
        if Path(name).name != name:
            raise ValueError("CPU plan must be a file inside the experiment")
        return HERE / name
    return HERE / ("flat_mapping_host_result.json" if entry.get("kernel_policy") == "flat_output_tiles"
                   else "distributed_write_host_result.json")


def _supplied_args(entry: dict[str, Any]) -> list[str]:
    result = ["--tiling-words", str(_supplied_plan(entry))]
    if entry.get("expected_supplied_outputs", 4) == 8:
        result += ["--value-offsets", "0", "1"]
    return result


def _boundary_source(text: str, entry: dict[str, Any], no_admission: bool = False) -> str:
    if text.count("190 * 1024**2") != 1:
        raise ValueError("Unexpected boundary-test source; inspect before adapting")
    text = text.replace("190 * 1024**2", str(entry["budget_mib"]) + " * 1024**2")
    if entry.get("policy", "read_balance") == "input_priority":
        if not no_admission:
            text = text.replace("4, 128, 1024, 65", "4, 128, 1023, 65")
        text = text.replace("row_bytes = 2 * length * dim", "row_bytes = 2 * (length + 1) * dim")
        text = text.replace("2 * query_elements * (2 * length + 5)", "4 * query_elements * (length + 2)")
        text = text.replace("group = 48 // math.gcd(history_tiles, 48)",
            "group = 48 // math.gcd(history_tiles, 48)\n"
            "    alignment_group = 256 // math.gcd((length + 1) * dim, 256)\n"
            "    group = math.lcm(group, alignment_group)")
        text = text.replace("if cache_rows >= group:\n        cache_rows -= cache_rows % group",
                            "cache_rows -= cache_rows % group")
        text = text.replace("normal_value_rows=cache_rows", "normal_key_rows=cache_rows")
        text = text.replace("input_boundary_mod32=(cache_rows * row_bytes + 2 * offset) % 32",
                            "output_boundary_mod512=(cache_rows * row_bytes) % 512")
        text = text.replace("Value-input", "Key-output")
        if no_admission:
            text = text.replace("if not 0 < cache_rows < batch * heads:", "if cache_rows != 0:")
            text = text.replace("case must cross the Key-output cache admission boundary",
                                "case must admit zero Key-output rows")
    elif entry.get("kernel_policy") in ("key_chain_only", "outputs_only"):
        begin = text.index("    row_bytes =")
        end = text.index("    shapes =", begin)
        text = text[:begin] + "    cache_rows = 0\n" + text[end:]
        text = text.replace("input_boundary_mod32=(cache_rows * row_bytes + 2 * offset) % 32", "value_input_bypass=True, key_output_normal=True")
        text = text.replace("across the Value-input admission boundary", "under the two-plane cache plan")
    elif entry.get("policy") == "input_only":
        begin = text.index("    row_bytes =")
        end = text.index("    shapes =", begin)
        text = text[:begin] + "    cache_rows = 0\n" + text[end:]
        text = text.replace("normal_value_rows=cache_rows", "normal_key_output_rows=cache_rows")
        text = text.replace("input_boundary_mod32=(cache_rows * row_bytes + 2 * offset) % 32", "history_inputs_normal=True, all_outputs_bypass=True")
        text = text.replace("across the Value-input admission boundary", "under the input-only cache plan")
    elif no_admission:
        raise ValueError("No-admission boundary belongs only to the input-priority candidates")
    if entry.get("kernel_policy") == "mirror_key_value":
        text = text.replace("normal_value_rows=cache_rows", "normal_key_input_rows=cache_rows")
        text = text.replace("Value-input", "Key-input")
    if entry.get("kernel_policy") == "key_output_placement":
        # The task allows independent contiguous outputs with nonzero offset;
        # record the actual data/base difference, not only allocator block VAs.
        offset = entry["key_output_offset_elements"]
        text = text.replace("item.update(status=\"passed\", shape=",
            "if outputs[1].storage_offset() != " + str(offset) + ":\n"
            "                        raise AssertionError(\"unexpected Key output placement\")\n"
            "                    item.update(output_storage_offsets=[t.storage_offset() for t in outputs],\n"
            "                                output_data_addresses=[t.data_ptr() for t in outputs],\n"
            "                                output_storage_addresses=[t.untyped_storage().data_ptr() for t in outputs])\n"
            "                    item.update(status=\"passed\", shape=")
    if entry.get("kernel_policy") == "reversed_output_allocation":
        text = text.replace("item.update(status=\"passed\", shape=",
            "item.update(output_data_addresses=[t.data_ptr() for t in outputs],\n"
            "                                output_storage_addresses=[t.untyped_storage().data_ptr() for t in outputs])\n"
            "                    item.update(status=\"passed\", shape=")
    return text


def _distributed_source(text: str, entry: dict[str, Any]) -> str:
    derived = _boundary_source(text, entry).replace("4, 128, 1024, 65", "16, 16, 2048, 64")
    derived = derived.replace("normal_value_rows=cache_rows", "equivalent_normal_value_rows=cache_rows, value_tile_pattern=\"15-normal-1-bypass\"")
    derived = derived.replace("input_boundary_mod32=(cache_rows * row_bytes + 2 * offset) % 32", "tile_period=16, normal_value_tiles=720, bypass_value_tiles=48")
    derived = derived.replace("    qkv = arrays[0].reshape",
        "    for index in (1, 2):\n"
        "        arrays[index] ^= (np.arange(batch * heads, dtype=np.uint16).reshape(batch, heads, 1, 1) * np.uint16(257))\n"
        "    qkv = arrays[0].reshape")
    return derived


def _distributed_key_source(text: str) -> str:
    text = text.replace("4, 128, 1024, 65", "16, 16, 2048, 64")
    begin, end = text.index("    row_bytes ="), text.index("    shapes =")
    text = text[:begin] + "    cache_rows = 240\n" + text[end:]
    text = text.replace("normal_value_rows=cache_rows", "normal_key_rows=cache_rows")
    text = text.replace("input_boundary_mod32=(cache_rows * row_bytes + 2 * offset) % 32", "normal_key_regions=[[4,84],[88,168],[172,252]], class_alignment=512")
    text = text.replace("    qkv = arrays[0].reshape",
        "    for index in (1, 2):\n"
        "        arrays[index] ^= (np.arange(batch * heads, dtype=np.uint16).reshape(batch, heads, 1, 1) * np.uint16(257))\n"
        "    qkv = arrays[0].reshape")
    return text


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["control", "c1_186mib", "c2_182mib"])
    parser.add_argument("--full", action="store_true", help="Also run queue, tail and hot geometry checks")
    args = parser.parse_args()
    root, reports = args.prepared_root.resolve(), args.report_dir.resolve()
    package = root / "official_task_package"
    files = _freeze_files(root, args.candidates, package)
    reports.mkdir(parents=True, exist_ok=False)
    # Legacy boundary metadata assumed 190 MiB. Adapt only the expected boundary
    # calculation in a separately archived executed test, never the built tree.
    for name in args.candidates:
        entry = _candidate(root, name)
        text = (root / name / "project/tests/cache_boundary.py").read_text()
        path = reports / (name + "_executed_cache_boundary.py")
        path.write_text(_boundary_source(text, entry))
        files.append(str(path))
        if entry.get("policy") == "input_priority":
            path = reports / (name + "_executed_cache_no_admission.py")
            path.write_text(_boundary_source(text, entry, no_admission=True))
            files.append(str(path))
        if entry.get("kernel_policy") == "distributed_value_input":
            path = reports / (name + "_executed_distributed_value.py")
            derived = _distributed_source(text, entry)
            path.write_text(derived)
            files.append(str(path))
        if entry.get("kernel_policy") in ("distributed_key_output", "flat_output_tiles", "dispersed_key_output", "paired_history", "interleaved_history"):
            path = reports / (name + "_executed_distributed_key.py")
            derived = _distributed_key_source(text)
            if entry.get("kernel_policy") == "flat_output_tiles":
                derived = derived.replace("normal_key_regions=[[4,84],[88,168],[172,252]]", "normal_key_regions=[[0,240]], output_tile_elements=47104")
            if entry.get("kernel_policy") in ("dispersed_key_output", "paired_history", "interleaved_history"):
                derived = derived.replace("normal_key_rows=cache_rows", "legacy_key_quota_rows=cache_rows")
                derived = derived.replace("normal_key_regions=[[4,84],[88,168],[172,252]]",
                    "normal_write_slots=" + str(entry["normal_output_slots"]) + ", slot_period=" +
                    str(entry["output_slot_period"]) + ", output_tile_elements=47104")
                if entry.get("bypass_output_slots"):
                    period = "slot_period=" + str(entry["output_slot_period"]) + ","
                    derived = derived.replace(period, "bypass_slot_pattern=" + str(entry["bypass_output_slots"]) + ", " + period)
                if entry.get("kernel_policy") == "paired_history":
                    derived = derived.replace("output_tile_elements=47104", "parent_tile_elements=47104, output_plane_elements=" + str(entry["paired_plane_elements"]))
            path.write_text(derived)
            files += [str(path), str(_supplied_plan(entry))]
            if entry.get("kernel_policy") in ("paired_history", "interleaved_history"):
                files.append(str(HERE / "identity/check_source_layouts.py"))
        if entry.get("kernel_policy") == "four_plane":
            path = reports / (name + "_executed_four_plane.py")
            derived = _distributed_key_source(text)
            derived = derived.replace("cache_rows = 240", "cache_rows = " + str(entry["four_plane_rows"]))
            derived = derived.replace("normal_key_regions=[[4,84],[88,168],[172,252]]",
                                      "normal_rows_per_plane=cache_rows, planes=4")
            path.write_text(derived)
            files += [str(path), str(_supplied_plan(entry)), str(HERE / "identity/check_supplied_outputs.py")]
    if any(_candidate(root, n).get("kernel_policy") in ("distributed_key_output", "flat_output_tiles", "dispersed_key_output", "paired_history", "interleaved_history") for n in args.candidates):
        files += [str(HERE / "identity/check_supplied_outputs.py"), str(HERE / "distributed_write_host_result.json"), str(HERE / "flat_mapping_host_result.json")]
    before = _snapshot(files)
    _write(reports / "pretest_manifest.json", before)
    summary = {"scope": "functional_pre_screen" if not args.full else "functional_regression",
               "status": "running", "npu_executed": False, "records": []}
    _write(reports / "summary.json", summary)
    try:
        for name in args.candidates:
            project = root / name / "project"
            for allocator in ["default", "page1g"]:
                env = _environment(root, name, allocator)
                directory = reports / (name + "_" + allocator)
                directory.mkdir()
                identity = directory / "identity.json"
                rc = _child([sys.executable, str(HERE / "probe_runtime.py"), "--report", str(identity)],
                            project, env, directory / "identity.log")
                if rc != 0 or _read(identity).get("device_name") != "Ascend910_9382":
                    raise ValueError("A3 guard failed before functional execution")
                commands = [
                    ("public_accuracy", package / "verify.py", []),
                    ("extended", project / "tests/acceptance.py", ["--package-dir", str(package)]),
                    ("cache_boundary", reports / (name + "_executed_cache_boundary.py"), []),
                ]
                if _candidate(root, name).get("policy") == "input_priority":
                    commands.append(("cache_no_admission", reports / (name + "_executed_cache_no_admission.py"), []))
                if _candidate(root, name).get("kernel_policy") == "distributed_value_input":
                    commands.append(("distributed_value_boundary", reports / (name + "_executed_distributed_value.py"), []))
                if _candidate(root, name).get("kernel_policy") in ("distributed_key_output", "flat_output_tiles", "dispersed_key_output", "paired_history", "interleaved_history"):
                    commands += [
                        ("dispersed_key_boundary" if _candidate(root, name).get("kernel_policy") in ("dispersed_key_output", "paired_history", "interleaved_history") else "distributed_key_boundary", reports / (name + "_executed_distributed_key.py"), []),
                        ("supplied_outputs", HERE / "identity/check_supplied_outputs.py", _supplied_args(_candidate(root, name))),
                    ]
                if _candidate(root, name).get("kernel_policy") == "four_plane":
                    commands += [
                        ("four_plane_boundary", reports / (name + "_executed_four_plane.py"), []),
                        ("supplied_outputs", HERE / "identity/check_supplied_outputs.py", _supplied_args(_candidate(root, name))),
                    ]
                if _candidate(root, name).get("kernel_policy") in ("paired_history", "interleaved_history"):
                    commands.append(("source_layouts", HERE / "identity/check_source_layouts.py", []))
                if args.full:
                    commands += [
                        ("queue_stress", project / "tests/queue_stress.py", ["--package-dir", str(package)]),
                        ("tail_fusion_boundary", project / "tests/tail_fusion_boundary.py", []),
                        ("hot_geometry_boundary", project / "tests/hot_geometry_boundary.py", []),
                    ]
                for label, script, extra in commands:
                    if _snapshot(files) != before:
                        raise ValueError("Source/artifact changed before functional execution")
                    result_path = directory / (label + ".json")
                    command = [sys.executable, str(script), "--solution-file", str(project / "solution.py"),
                               "--report", str(result_path), *extra]
                    rc = _child(command, package, env, directory / (label + ".log"))
                    result = _read(result_path)
                    summary["npu_executed"] |= result.get("npu_executed") is True
                    record = {"candidate": name, "allocator": allocator, "test": label,
                              "exit_code": rc, "status": result.get("status"), "report": str(result_path)}
                    summary["records"].append(record)
                    _write(reports / "summary.json", summary)
                    if rc != 0 or result.get("status") != "passed" or result.get("npu_executed") is not True:
                        raise ValueError("Candidate functional check failed; inspect original records")
                    expected = {"public_accuracy": 35, "extended": 206, "cache_boundary": 8, "cache_no_admission": 8, "distributed_value_boundary": 8, "distributed_key_boundary": 8, "four_plane_boundary": 8, "supplied_outputs": 4}
                    expected["dispersed_key_boundary"] = 8
                    expected["source_layouts"] = 12
                    expected["supplied_outputs"] = _candidate(root, name).get("expected_supplied_outputs", 4)
                    if label in expected and len(result.get("cases", result.get("records", []))) != expected[label]:
                        raise ValueError("Unexpected functional record count")
        summary["status"] = "passed"
    except (ValueError, KeyError, TypeError, OSError, KeyboardInterrupt) as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        after = _snapshot(files)
        _write(reports / "posttest_manifest.json", after)
        summary["source_and_artifacts_unchanged"] = before == after
        if before != after:
            summary["status"] = "failed"
        _write(reports / "summary.json", summary)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
