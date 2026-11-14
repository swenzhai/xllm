# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Frozen old/new artifacts crossed with visible chips and allocator contexts."""

import argparse
import fcntl
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence import _child, _read, _write
from run_matrix import _evaluate, _snapshot

HERE = Path(__file__).resolve().parent


def _environment(config: dict[str, Any], version: str, allocator: str, device: str) -> dict[str, str]:
    item = config[version]
    env = dict(os.environ)
    prefix, project = Path(item["prefix"]), Path(item["project"])
    env.update(ASCEND_CUSTOM_OPP_PATH=str(prefix),
        T5QKV_OPAPI_LIB=str(prefix / "op_api/lib/libcust_opapi.so"), ASCEND_RT_VISIBLE_DEVICES=device)
    env["LD_LIBRARY_PATH"] = ":".join([str(prefix / "op_api/lib"),
        str(project / "build_direct"), env.get("LD_LIBRARY_PATH", "")])
    for key in ["PYTORCH_NPU_ALLOC_CONF", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING"]:
        env.pop(key, None)
    if allocator == "page1g":
        env["PYTORCH_NPU_ALLOC_CONF"] = "page_size:1g"
    elif allocator != "default":
        raise ValueError("Unknown allocator context")
    return env


def _identity(path: Path, config: dict[str, Any], version: str) -> dict[str, Any]:
    identity = _read(path)
    if identity["device_name"] != "Ascend910_9382" or identity["visible_devices"] != 1:
        raise ValueError("Wrong chip or number of visible devices")
    item = config[version]
    project, prefix = Path(item["project"]), Path(item["prefix"])
    required = [next(project.glob("_t5qkv*.so")), project / "build_direct/libt5qkv_direct.so",
                prefix / "op_api/lib/libcust_opapi.so",
                prefix / "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
                prefix / "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"]
    expected = _snapshot([str(p) for p in required])
    if any(identity["loaded_files"].get(p) != h for p, h in expected.items()):
        raise ValueError("Requested libraries are not all mapped at their frozen identities")
    unexpected = [p for p in identity["loaded_files"] if ("/project" in p or "/install" in p)
                  and "t5qkv" in p and p not in expected]
    if unexpected:
        raise ValueError("Unexpected operator library mapping: " + str(unexpected))
    return identity


def _execute(config: dict[str, Any], reports: Path, audit: Path, config_path: Path) -> int:
    audit_data = _read(audit)
    package = Path(config["package"])
    new = config["new"]
    directory = Path(new["project"]).parent
    build = _read(Path(new["build_manifest"]))
    prepared = _read(directory.parent / "prepared_manifest.json")
    sources = _read(Path(new["source_manifest"]))["source_files"]
    paths = [config_path, audit, Path(new["build_manifest"]), Path(new["source_manifest"])]
    paths += [Path(p) for p in config["old"]["expected_artifacts"]]
    paths += [p for p in Path(config["old"]["prefix"]).rglob("*") if p.is_file()]
    paths += [directory / p for p in build["artifacts"]]
    paths += [Path(config[v]["project"]) / p for v in ["old", "new"] for p in sources]
    paths += [package / p for p in prepared["official_files"]]
    paths += [p for p in HERE.rglob("*") if p.suffix in (".py", ".json")]
    paths += [HERE.parent / n for n in ["evidence.py", "run_matrix.py"]]
    before = _snapshot([str(p) for p in paths])
    if any(before[str(package / p)] != h for p, h in prepared["official_files"].items()):
        raise ValueError("Official package differs from pinned source")
    for version in ["old", "new"]:
        for name, record in audit_data["artifacts"][version].items():
            if before.get(record["path"]) != record["sha256"]:
                raise ValueError("Artifact changed since ELF audit: " + name)
    reports.mkdir(parents=True, exist_ok=False)
    _write(reports / "pretest_manifest.json", before)
    result: dict[str, Any] = {"scope": "binary_device_allocator_crossover_not_final_acceptance",
        "status": "running", "records": [], "precision": [], "npu_executed": False}
    _write(reports / "summary.json", result)
    try:
        for device in config["visible_devices"]:
            for allocator in ["default", "page1g"]:
                for version in ["old", "new"]:
                    item = reports / f"precision_{device}_{allocator}_{version}"
                    item.mkdir()
                    env = _environment(config, version, allocator, device)
                    command = [sys.executable, str(HERE / "after_benchmark.py"),
                        "--script", str(package / "verify.py"), "--identity", str(item / "identity.json"),
                        "--expected-prefix", config[version]["prefix"], "--", "--solution-file",
                        str(Path(config[version]["project"]) / "solution.py"), "--report", str(item / "verify.json")]
                    rc = _child(command, package, env, item / "verify.log")
                    identity = _identity(item / "identity.json", config, version)
                    original = _read(item / "verify.json")
                    if rc != 0 or original["status"] != "passed" or original.get("npu_executed") is not True or len(original["cases"]) != 35:
                        raise ValueError("Precision precondition failed")
                    result["precision"].append({"device": device, "allocator": allocator,
                        "version": version, "status": "passed", "cases": 35, "pid": identity["pid"]})
                    _write(reports / "summary.json", result)
        targets = _read(package / "performance_targets.json")
        schedule = []
        for batch in range(config["batches"]):
            devices = config["visible_devices"] if batch % 2 == 0 else list(reversed(config["visible_devices"]))
            for device in devices:
                for allocator in (["default", "page1g"] if batch % 2 == 0 else ["page1g", "default"]):
                    versions = ["old", "new"] if (batch + int(device)) % 2 == 0 else ["new", "old"]
                    schedule += [(batch, device, allocator, version) for version in versions]
        for index, (batch, device, allocator, version) in enumerate(schedule):
            if _snapshot(list(before)) != before:
                raise ValueError("Identity changed before benchmark")
            item = reports / f"{index:03}_{version}_{allocator}_chip{device}"
            item.mkdir()
            command = [sys.executable, str(HERE / "after_benchmark.py"),
                "--script", str(package / "benchmark.py"), "--identity", str(item / "identity.json"),
                "--expected-prefix", config[version]["prefix"], "--", "--solution-file",
                str(Path(config[version]["project"]) / "solution.py"), "--device", "0",
                "--report", str(item / "benchmark.json")]
            rc = _child(command, package, _environment(config, version, allocator, device), item / "benchmark.log")
            identity = _identity(item / "identity.json", config, version)
            checked = _evaluate(_read(item / "benchmark.json"), targets)
            if rc != (0 if checked["status"] == "passed" else 1):
                raise ValueError("Benchmark exit disagrees with original checker")
            result["npu_executed"] = True
            result["records"].append({"batch": batch, "visible_device": device, "allocator": allocator,
                "version": version, "pid": identity["pid"], "report": str(item / "benchmark.json"), **checked})
            _write(reports / "summary.json", result)
        result["status"] = "completed_diagnostic_crossover"
    except (OSError, ValueError, KeyError, TypeError, KeyboardInterrupt) as error:
        result.update(status="invalid_or_incomplete", error=f"{type(error).__name__}: {error}")
    finally:
        after = _snapshot(list(before))
        _write(reports / "posttest_manifest.json", after)
        result["source_and_artifacts_unchanged"] = before == after
        if before != after:
            result["status"] = "invalid_or_incomplete"
        _write(reports / "summary.json", result)
    return 0 if result["status"] == "completed_diagnostic_crossover" else 1


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    with (args.report_dir.parent / "crossover.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute(_read(args.config), args.report_dir.resolve(), args.audit.resolve(), args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(_main())
