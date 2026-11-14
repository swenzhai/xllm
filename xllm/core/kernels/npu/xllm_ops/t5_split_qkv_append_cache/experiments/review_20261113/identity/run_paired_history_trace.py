# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Diagnose paired-read eligibility with full allocation history, never acceptance."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import sys
import sysconfig
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence import _child, _environment, _read, _write
from run_matrix import _evaluate, _freeze_files, _snapshot
from identity.run_crossover import _identity

HERE = Path(__file__).resolve().parent


def _address_rows(cases: list[dict[str, Any]], allocs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected, spans = [], []
    for case in cases:
        inputs = [math.prod(d["shape"]) * 2 for d in case["input_desc"]]
        inputs = [size for size in inputs if size]
        outputs = [math.prod(d["shape"]) * 2 for d in case["output_desc"]]
        spans.append((len(expected), len(inputs)))
        expected += inputs + outputs * 1051
    if [e["size"] for e in allocs] != expected:
        raise ValueError("Allocation history does not match every input and output of the original protocol")
    rows = []
    for case, (begin, count) in zip(cases, spans):
        if "_007_" not in case["case_name"] and "_017_" not in case["case_name"]:
            continue
        inputs = [e["addr"] for e in allocs[begin:begin + count]]
        first = begin + count + 51 * 3
        triples = Counter(tuple(e["addr"] for e in allocs[i:i + 3])
                          for i in range(first, first + 3000, 3))
        gap = inputs[2] - inputs[1]
        rows.append({"case_name": case["case_name"], "input_addresses": inputs,
                     "value_minus_key_bytes": gap,
                     "timed_output_triples": [
                         {"addresses": list(t), "count": n,
                          "c27_address_guard_expected":
                              47104 <= gap <= 0xffffffff and (inputs[1] | inputs[2]) % 2 == 0
                              and t[1] % 512 == 0 and t[2] % 512 == 0}
                         for t, n in triples.items()],
                     "branch_observed_on_device": False})
    return rows


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--candidate", default="c27_paired_history_pad")
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--allocator", choices=["default", "page1g"], default="default")
    parser.add_argument("--metric", choices=["L2Cache", "PipeUtilization"])
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Require at least one complete diagnostic process")
    root, reports = args.prepared_root.resolve(), args.report_dir.resolve()
    package = root / "official_task_package"
    files = _freeze_files(root, [args.candidate], package)
    files += [str(Path(__file__).resolve()), str(HERE / "memory_history.py"),
              str(HERE / "run_crossover.py")]
    before = _snapshot(files)
    cases = [c for c in _read(package / "case.json") if c.get("performance") is True]
    reports.mkdir(parents=True, exist_ok=False)
    _write(reports / "pretest_manifest.json", before)
    result: dict[str, Any] = {"scope": "paired_read_address_diagnostic_NOT_ACCEPTANCE",
                              "status": "running", "records": [], "full_acceptance": False}
    _write(reports / "summary.json", result)
    try:
        for index in range(args.repetitions):
            if _snapshot(files) != before:
                raise ValueError("Frozen source or artifact changed")
            directory = reports / f"{args.allocator}_{index:02}"
            directory.mkdir()
            trace = directory / "memory_trace.json.gz"
            identity = directory / "identity.json"
            command = [sys.executable, str(HERE / "memory_history.py"),
                       "--site-packages", sysconfig.get_path("purelib"),
                       "--benchmark", str(package / "benchmark.py"),
                       "--trace", str(trace), "--identity", str(identity), "--",
                       "--solution-file", str(root / args.candidate / "project/solution.py"),
                       "--device", "0", "--report", str(directory / "benchmark.json")]
            if args.metric:
                profiler = shutil.which("msprof")
                if profiler is None:
                    raise ValueError("CANN msprof is required for requested diagnostic counters")
                command = [profiler, "--application=" + shlex.join(command),
                           "--output=" + str(directory / "profile"),
                           "--aic-metrics=" + args.metric]
                if args.metric == "L2Cache":
                    command.append("--l2=on")
            rc = _child(command, package, _environment(root, args.candidate, args.allocator),
                        directory / "benchmark.log")
            observed = _identity(identity, {"new": {
                "project": str(root / args.candidate / "project"),
                "prefix": str(root / args.candidate / "install/vendors/t5qkv")}}, "new")
            if hashlib.sha256(trace.read_bytes()).hexdigest() != observed["trace_sha256"]:
                raise ValueError("Memory trace identity mismatch")
            checked = _evaluate(_read(directory / "benchmark.json"),
                                _read(package / "performance_targets.json"))
            original_rc = observed["original_exit_code"]
            if original_rc != (0 if checked["status"] == "passed" else 1):
                raise ValueError("Original benchmark exit code mismatch")
            csv = list(directory.rglob("op_summary*.csv"))
            if args.metric and (rc != 0 or len(csv) != 1):
                raise ValueError("Incomplete profiler diagnostic")
            if not args.metric and rc != original_rc:
                raise ValueError("Diagnostic wrapper exit code mismatch")
            with gzip.open(trace, "rt") as stream:
                snapshot = json.load(stream)
            events = [e for seq in snapshot["device_traces"] for e in seq]
            if len(events) >= 200000:
                raise ValueError("Truncated history")
            allocs = [e for e in events if e["action"] == "alloc"]
            result["records"].append({"directory": str(directory), "allocations": len(allocs),
                                      "metric": args.metric, "allocator": args.allocator,
                                      "profiler_csv": [str(p) for p in csv],
                                      "events": len(events), "address_rows": _address_rows(cases, allocs),
                                      "instrumented_checks_NOT_ACCEPTANCE": checked})
            _write(reports / "summary.json", result)
        result["status"] = "completed_diagnostic_NOT_ACCEPTANCE"
    except (ValueError, KeyError, TypeError, OSError, KeyboardInterrupt) as error:
        result.update(status="invalid_or_incomplete", error=f"{type(error).__name__}: {error}")
    finally:
        after = _snapshot(files)
        _write(reports / "posttest_manifest.json", after)
        result["source_and_artifacts_unchanged"] = before == after
        if before != after:
            result["status"] = "invalid_or_incomplete"
        _write(reports / "summary.json", result)
    return 0 if result["status"] == "completed_diagnostic_NOT_ACCEPTANCE" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
