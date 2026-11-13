# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Collect diagnostic PMU evidence separately from uninstrumented acceptance."""

import argparse
from pathlib import Path
import shlex
import shutil
import sys
import sysconfig

from evidence import _child, _environment, _read, _write
from run_matrix import _evaluate, _freeze_files, _snapshot

HERE = Path(__file__).resolve().parent


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--candidate", default="control")
    args = parser.parse_args()
    root, reports = args.prepared_root.resolve(), args.report_dir.resolve()
    package = root / "official_task_package"
    files = _freeze_files(root, [args.candidate], package)
    profiler = shutil.which("msprof")
    if profiler is None:
        parser.error("CANN msprof is required")
    reports.mkdir(parents=True, exist_ok=False)
    before = _snapshot(files)
    _write(reports / "pretest_manifest.json", before)
    summary = {"scope": "instrumented_profile_diagnostic_only", "status": "running",
               "npu_executed": False, "formal_acceptance": False, "records": []}
    _write(reports / "summary.json", summary)
    try:
        for allocator in ["default", "page1g"]:
            env = _environment(root, args.candidate, allocator)
            identity = reports / (allocator + "_identity.json")
            rc = _child([sys.executable, str(HERE / "probe_runtime.py"), "--report", str(identity)],
                        package, env, reports / (allocator + "_identity.log"))
            if rc != 0 or _read(identity).get("device_name") != "Ascend910_9382":
                raise ValueError("Target A3 identity guard failed")
            for metric in ["PipeUtilization", "L2Cache"]:
                if _snapshot(files) != before:
                    raise ValueError("Source/artifact changed before profiling")
                directory = reports / (allocator + "_" + metric)
                directory.mkdir()
                result_path = directory / "benchmark.json"
                application = [sys.executable, str(HERE / "profile_application.py"),
                               "--site-packages", sysconfig.get_path("purelib"),
                               "--identity", str(directory / "application_identity.json"),
                               "--benchmark", str(package / "benchmark.py"), "--",
                               "--solution-file", str(root / args.candidate / "project/solution.py"),
                               "--device", "0", "--report", str(result_path)]
                command = [profiler, "--application=" + shlex.join(application),
                           "--output=" + str(directory / "profile"), "--aic-metrics=" + metric]
                if metric == "L2Cache":
                    command.append("--l2=on")
                rc = _child(command, package, env, directory / "profile.log")
                original = _read(result_path)
                checked = _evaluate(original, _read(package / "performance_targets.json"))
                summary["npu_executed"] |= original["npu_executed"]
                csv_files = sorted(str(p.relative_to(directory)) for p in directory.rglob("*.csv"))
                if not csv_files:
                    raise ValueError("No profiler CSV exported; do not infer counters from stdout")
                summary["records"].append({"allocator": allocator, "metric": metric,
                    "profiler_exit_code": rc, "profile_directory": str(directory),
                    "csv_files": csv_files, "instrumented_absolute_checks": checked,
                    "status": "collected_not_acceptance" if rc == 0 else "requires_profiler_error_review"})
                _write(reports / "summary.json", summary)
        summary["status"] = "collected_not_acceptance" if all(
            r["profiler_exit_code"] == 0 for r in summary["records"]) else "requires_profiler_error_review"
    except (ValueError, OSError, KeyError, TypeError, KeyboardInterrupt) as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        after = _snapshot(files)
        _write(reports / "posttest_manifest.json", after)
        summary["source_and_artifacts_unchanged"] = before == after
        if before != after:
            summary["status"] = "failed"
        _write(reports / "summary.json", summary)
    return 0 if summary["status"] == "collected_not_acceptance" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
