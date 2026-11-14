# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Collect allocator history and L2 counters in the same diagnostic process."""

import argparse
from pathlib import Path
import shlex
import shutil
import sys
import sysconfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence import _child, _read, _write
from run_matrix import _evaluate, _freeze_files, _snapshot
from run_crossover import _environment, _identity

HERE = Path(__file__).resolve().parent


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    config, reports = _read(args.config), args.report_dir.resolve()
    root = Path(config["new"]["project"]).parent.parent
    package = Path(config["package"])
    files = _freeze_files(root, ["control"], package)
    files += [str(p) for p in HERE.iterdir() if p.suffix in (".py", ".json")]
    profiler = shutil.which("msprof")
    if profiler is None:
        parser.error("CANN msprof is required")
    reports.mkdir(parents=True, exist_ok=False)
    before = _snapshot(files)
    _write(reports / "pretest_manifest.json", before)
    result = {"scope": "same_process_allocation_history_and_L2_diagnostic_not_acceptance",
              "status": "running", "records": []}
    _write(reports / "summary.json", result)
    try:
        for device in config["visible_devices"]:
            for allocator in ["default", "page1g"]:
                if _snapshot(files) != before:
                    raise ValueError("Identity changed before profiling")
                item = reports / f"chip{device}_{allocator}"
                item.mkdir()
                application = [sys.executable, str(HERE / "memory_history.py"),
                    "--site-packages", sysconfig.get_path("purelib"),
                    "--benchmark", str(package / "benchmark.py"),
                    "--trace", str(item / "memory_trace.json.gz"),
                    "--identity", str(item / "identity.json"), "--", "--solution-file",
                    str(Path(config["new"]["project"]) / "solution.py"),
                    "--device", "0", "--report", str(item / "benchmark.json")]
                command = [profiler, "--application=" + shlex.join(application),
                    "--output=" + str(item / "profile"), "--aic-metrics=L2Cache", "--l2=on"]
                rc = _child(command, package, _environment(config, "new", allocator, device), item / "profile.log")
                identity = _identity(item / "identity.json", config, "new")
                checked = _evaluate(_read(item / "benchmark.json"), _read(package / "performance_targets.json"))
                csv = list(item.rglob("op_summary*.csv"))
                if rc != 0 or len(csv) != 1 or not (item / "memory_trace.json.gz").is_file():
                    raise ValueError("Incomplete profiler/history collection")
                result["records"].append({"visible_device": device, "allocator": allocator,
                    "profiler_exit_code": rc, "pid": identity["pid"], "directory": str(item),
                    "csv": str(csv[0]), "instrumented_checks_not_acceptance": checked})
                _write(reports / "summary.json", result)
        result["status"] = "collected_not_acceptance"
    except (OSError, ValueError, KeyError, TypeError, KeyboardInterrupt) as error:
        result.update(status="invalid_or_incomplete", error=f"{type(error).__name__}: {error}")
    finally:
        after = _snapshot(files)
        _write(reports / "posttest_manifest.json", after)
        result["source_and_artifacts_unchanged"] = before == after
        if before != after:
            result["status"] = "invalid_or_incomplete"
        _write(reports / "summary.json", result)
    return 0 if result["status"] == "collected_not_acceptance" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
