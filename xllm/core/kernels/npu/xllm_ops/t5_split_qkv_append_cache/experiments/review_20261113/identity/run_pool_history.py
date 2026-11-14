# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Compare complete allocation VAs with and without conditioning; never acceptance."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
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
    root, package = Path(config["new"]["project"]).parent.parent, Path(config["package"])
    files = _freeze_files(root, ["control"], package)
    files += [str(p) for p in HERE.iterdir() if p.suffix in (".py", ".json")]
    before = _snapshot(files)
    cases = [c for c in _read(package / "case.json") if c.get("performance") is True]
    expected, spans = [], []
    for case in cases:
        sizes = [math.prod(d["shape"]) * 2 for d in case["input_desc"]]
        inputs = [size for size in sizes if size]
        outputs = [math.prod(d["shape"]) * 2 for d in case["output_desc"]]
        spans.append((len(expected), len(inputs)))
        expected += inputs + outputs * 1051
    reports.mkdir(parents=True, exist_ok=False)
    _write(reports / "pretest_manifest.json", before)
    result = {"scope": "allocator_history_conditioning_diagnostic_NOT_ACCEPTANCE",
              "status": "running", "records": [], "full_acceptance": False}
    _write(reports / "summary.json", result)
    try:
        for device in config["visible_devices"]:
            for allocator, primer in [("default", 0), ("default", 1073741312), ("page1g", 0)]:
                if _snapshot(files) != before:
                    raise ValueError("Frozen source changed")
                item = reports / f"chip{device}_{allocator}_{primer}"
                item.mkdir()
                command = [sys.executable, str(HERE / "memory_history.py"),
                    "--site-packages", sysconfig.get_path("purelib"), "--primer-bytes", str(primer),
                    "--benchmark", str(package / "benchmark.py"), "--trace", str(item / "memory_trace.json.gz"),
                    "--identity", str(item / "identity.json"), "--", "--solution-file",
                    str(Path(config["new"]["project"]) / "solution.py"), "--device", "0", "--report", str(item / "benchmark.json")]
                rc = _child(command, package, _environment(config, "new", allocator, device), item / "benchmark.log")
                identity = _identity(item / "identity.json", config, "new")
                checked = _evaluate(_read(item / "benchmark.json"), _read(package / "performance_targets.json"))
                if rc != (0 if checked["status"] == "passed" else 1):
                    raise ValueError("Original benchmark exit mismatch")
                trace_path = item / "memory_trace.json.gz"
                if hashlib.sha256(trace_path.read_bytes()).hexdigest() != identity["trace_sha256"]:
                    raise ValueError("Trace hash mismatch")
                with gzip.open(trace_path, "rt") as stream:
                    trace = json.load(stream)
                events = [e for seq in trace["device_traces"] for e in seq]
                allocs = [e for e in events if e["action"] == "alloc"]
                if len(events) >= 200000 or [e["size"] for e in allocs] != expected:
                    raise ValueError("Full original allocation sequence differs or is truncated")
                rows = []
                for case, (begin, inputs) in zip(cases, spans):
                    if "_007_" not in case["case_name"] and "_017_" not in case["case_name"]:
                        continue
                    start = begin + inputs + 51 * 3
                    triples = Counter(tuple(e["addr"] for e in allocs[i:i + 3]) for i in range(start, start + 1000 * 3, 3))
                    rows.append({"case_name": case["case_name"], "input_addresses": [e["addr"] for e in allocs[begin:begin + inputs]],
                        "timed_output_triples": [{"addresses": list(t), "count": n} for t, n in triples.items()]})
                result["records"].append({"visible_device": device, "allocator": allocator, "primer_bytes": primer,
                    "directory": str(item), "allocations": len(allocs), "events": len(events), "address_rows": rows,
                    "segments": [{k: s[k] for k in ["address", "total_size", "segment_type"]} for s in trace["segments"]],
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
