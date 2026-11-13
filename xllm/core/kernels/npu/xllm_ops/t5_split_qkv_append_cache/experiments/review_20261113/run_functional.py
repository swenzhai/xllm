# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Pre-screen candidate precision/engineering/boundaries with frozen artifacts."""

import argparse
from pathlib import Path
import sys

from evidence import _candidate, _child, _environment, _read, _write
from run_matrix import _freeze_files, _snapshot

HERE = Path(__file__).resolve().parent


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
        if text.count("190 * 1024**2") != 1:
            raise ValueError("Unexpected boundary-test source; inspect before adapting")
        path = reports / (name + "_executed_cache_boundary.py")
        path.write_text(text.replace("190 * 1024**2", str(entry["budget_mib"]) + " * 1024**2"))
        files.append(str(path))
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
                    expected = {"public_accuracy": 35, "extended": 206, "cache_boundary": 8}
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
