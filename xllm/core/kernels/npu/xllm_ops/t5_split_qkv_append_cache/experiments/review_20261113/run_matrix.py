# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Run untouched public benchmarks in fresh processes; retain every failed batch.

Plan-only mode needs no CANN/torch/NPU. Actual execution requires the target A3.
This produces performance-only evidence, never a complete acceptance decision.
"""

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from evidence import _candidate, _check, _child, _environment, _read, _write

HERE = Path(__file__).resolve().parent


def _schedule(names: list[str], batches: int) -> list[dict[str, Any]]:
    result = []
    for batch in range(batches):
        rotated = names[batch % len(names):] + names[:batch % len(names)]
        allocators = ["default", "page1g"] if batch % 2 == 0 else ["page1g", "default"]
        for index, allocator in enumerate(allocators):
            order = rotated if index == 0 else list(reversed(rotated))
            result.extend({"batch": batch, "allocator": allocator, "candidate": name}
                          for name in order)
    return result


def _samples(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("Expected all five original round means")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in value):
        raise ValueError("Samples must be positive finite numbers")
    return value


def _evaluate(report: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    """Independently recompute, without changing original report status or bytes."""
    if report.get("error") or report.get("npu_executed") is not True:
        raise ValueError("Timing requires a complete NPU-executed report without errors")
    if report.get("protocol") != {"warmup": 50, "iterations": 200, "rounds": 5}:
        raise ValueError("Official timing protocol changed")
    required = {c["case_name"]: c for c in targets["cases"]}
    rows = report.get("cases", [])
    actual = {c["case_name"]: c for c in rows}
    if len(required) != 10 or len(rows) != 10 or set(actual) != set(required):
        raise ValueError("Missing, duplicated or extra performance cases")
    checks = []
    for name, target in required.items():
        row = actual[name]
        accuracy = row["accuracy"]
        if (len(accuracy) != 3 or {a["output"] for a in accuracy} != {"query", "key", "value"}
                or any(a["bit_mismatches"] != 0 for a in accuracy)):
            raise ValueError("A timed case did not preserve all three outputs bit-for-bit")
        measured = row["results"]["custom"]
        samples = _samples(measured["wall_us_rounds"])
        events = _samples(measured["device_event_us_rounds"])
        median = statistics.median(samples)
        if (median != measured["wall_us_median"]
                or statistics.median(events) != measured["device_event_us_median"]):
            raise ValueError("Cached median differs from the original five samples")
        checks.append({"case_name": name, "status": "passed" if median <= target["max_wall_us"] else "failed",
                       "wall_us_rounds": samples, "wall_us_median": median,
                       "max_wall_us": target["max_wall_us"]})
    status = "passed" if all(c["status"] == "passed" for c in checks) else "failed"
    original = report.get("performance_checks", {})
    if (report.get("status") != status or original.get("status") != status
            or original.get("checks") != checks):
        raise ValueError("Independent recomputation disagrees with the original official checker")
    return {"status": status, "checks": checks}


def _snapshot(paths: list[str]) -> dict[str, str]:
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file()
            else "MISSING" for p in paths}


def _freeze_files(root: Path, names: list[str], package: Path) -> list[str]:
    prepared = _read(root / "prepared_manifest.json")
    _check(package, prepared["official_files"])
    paths = [root / "prepared_manifest.json"]
    paths += [package / n for n in prepared["official_files"]]
    paths += [p for p in HERE.iterdir() if p.suffix in (".py", ".json", ".cpp")]
    for name in names:
        entry = _candidate(root, name)
        directory = root / name
        build = _read(directory / "build_manifest.json")
        if build.get("status") != "built_a3_unvalidated" or build.get("source_files") != entry["source_files"]:
            raise ValueError("Build has not completed against the exact candidate source")
        _check(directory, build["artifacts"])
        paths += [directory / "source_manifest.json", directory / "build_manifest.json",
                  directory / "candidate.patch"]
        paths += [directory / "project" / n for n in entry["source_files"]]
        paths += [directory / n for n in build["artifacts"]]
    return sorted({str(p) for p in paths})


def _benchmark(root: Path, package: Path, directory: Path, python: str,
               name: str, allocator: str, baseline: bool,
               targets: dict[str, Any]) -> dict[str, Any]:
    directory.mkdir(exist_ok=False)
    env = _environment(root, name, allocator)
    solution = root / name / "project" / ("tests/taskbook_baseline.py" if baseline else "solution.py")
    record: dict[str, Any] = {"candidate": name, "allocator": allocator,
                              "role": "reference" if baseline else "custom",
                              "report": str(directory / "benchmark.json"), "status": "error",
                              "npu_executed": False}
    command = [python, str(package / "benchmark.py"), "--solution-file", str(solution),
               "--device", "0", "--report", record["report"]]
    rc = _child(command, package, env, directory / "benchmark.log")
    record["exit_code"] = rc
    try:
        original = _read(Path(record["report"]))
        record["npu_executed"] = original.get("npu_executed") is True
        result = _evaluate(original, targets)
        if rc != (0 if result["status"] == "passed" else 1):
            raise ValueError("Child exit code disagrees with the measured status")
        record.update(result)
    except (ValueError, OSError, KeyError, TypeError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    _write(directory / "derived_check.json", record)
    return record


def _execute(root: Path, package: Path, directory: Path, names: list[str],
             plan: list[dict[str, Any]], python: str, phase: str) -> int:
    targets = _read(package / "performance_targets.json")
    files = _freeze_files(root, names, package)
    before = _snapshot(files)
    _write(directory / "pretest_manifest.json", before)
    summary: dict[str, Any] = {"scope": "performance_only", "phase": phase,
                               "status": "running", "npu_executed": False,
                               "records": [], "references": [], "full_acceptance": False}
    _write(directory / "summary.json", summary)
    try:
        # Guard both fresh-process allocator contexts before any timed workload.
        for allocator in ["default", "page1g"]:
            env = _environment(root, names[0], allocator)
            identity = directory / ("runtime_" + allocator + ".json")
            rc = _child([python, str(HERE / "probe_runtime.py"), "--report", str(identity)],
                        package, env, directory / ("runtime_" + allocator + ".log"))
            if (rc != 0 or _read(identity).get("device_name") != "Ascend910_9382"
                    or _read(identity).get("visible_devices") != 1):
                raise ValueError("Target A3 identity guard failed; no benchmark was launched")
            qualified = directory / ("reference_accuracy_" + allocator + ".json")
            reference = root / names[0] / "project/tests/taskbook_baseline.py"
            rc = _child([python, str(package / "verify.py"), "--solution-file", str(reference),
                         "--report", str(qualified)], package, env,
                        directory / ("reference_accuracy_" + allocator + ".log"))
            accuracy = _read(qualified)
            if (rc != 0 or accuracy.get("status") != "passed"
                    or accuracy.get("npu_executed") is not True or len(accuracy.get("cases", [])) != 35):
                raise ValueError("Reference did not qualify on all 35 public cases")
            result = _benchmark(root, package, directory / ("reference_" + allocator),
                                python, names[0], allocator, True, targets)
            # Reference threshold failure is preserved; it is not custom acceptance.
            if result["status"] == "error":
                raise ValueError("Invalid reference timing evidence")
            summary["references"].append(result)
            _write(directory / "summary.json", summary)
        for index, item in enumerate(plan):
            if _snapshot(files) != before:
                raise ValueError("Source/artifact changed before a benchmark")
            result = _benchmark(root, package, directory / f"{index:03}_{item['candidate']}_{item['allocator']}",
                                python, item["candidate"], item["allocator"], False, targets)
            result["batch"] = item["batch"]
            summary["records"].append(result)
            summary["npu_executed"] |= result["npu_executed"]
            _write(directory / "summary.json", summary)
        passed = (len(summary["records"]) == len(plan)
                  and all(r["status"] == "passed" for r in summary["records"]))
        summary["status"] = "passed" if passed else "failed"
    except (ValueError, OSError, KeyError, TypeError, KeyboardInterrupt) as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        after = _snapshot(files)
        _write(directory / "posttest_manifest.json", after)
        summary["source_and_artifacts_unchanged"] = before == after
        if before != after:
            summary["status"] = "failed"
        _write(directory / "summary.json", summary)
    return 0 if summary["status"] == "passed" else 1


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["control", "c1_186mib", "c2_182mib"])
    parser.add_argument("--phase", choices=["screen", "confirm"], default="screen")
    parser.add_argument("--batches", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    batches = args.batches if args.batches is not None else (10 if args.phase == "confirm" else 3)
    if (batches < 1 or len(set(args.candidates)) != len(args.candidates)
            or (args.phase == "confirm" and (batches < 10 or len(args.candidates) != 1))):
        parser.error("Confirmation requires one candidate and at least 10 batches per allocator")
    root = args.prepared_root.resolve()
    prepared = _read(root / "prepared_manifest.json")
    for name in args.candidates:
        _candidate(root, name)
    package = root / "official_task_package"
    _check(package, prepared["official_files"])
    plan = _schedule(args.candidates, batches)
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=False)
    _write(report / "execution_plan.json", {"status": "planned", "npu_executed": False,
            "phase": args.phase, "protocol": {"warmup": 50, "iterations": 200, "rounds": 5},
            "source_commit": prepared["source_commit"], "plan": plan})
    if args.plan_only:
        print(json.dumps({"status": "planned_not_executed", "npu_executed": False,
                          "custom_processes": len(plan), "report_dir": str(report)}))
        return 0
    with (root / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute(root, package, report, args.candidates, plan, args.python, args.phase)


if __name__ == "__main__":
    raise SystemExit(_main())
