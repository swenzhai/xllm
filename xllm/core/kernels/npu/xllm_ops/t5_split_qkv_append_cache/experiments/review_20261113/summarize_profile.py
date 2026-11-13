# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Derive per-case PMU summaries only when the full public task mapping matches."""

import argparse
import csv
import hashlib
import math
from pathlib import Path
import statistics
from typing import Any

from evidence import _read, _write


def _median(rows: list[dict[str, str]], field: str) -> float | None:
    values = []
    for row in rows:
        text = row.get(field, "N/A")
        if text in ("N/A", "", None):
            return None
        value = float(text)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid profiler value in {field}")
        values.append(value)
    return statistics.median(values)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new derived summary; preserve prior evidence")
    root = args.profile_root.resolve()
    collection = _read(root / "summary.json")
    if (collection["status"] != "collected_not_acceptance"
            or not collection["source_and_artifacts_unchanged"]):
        raise ValueError("Require a successful complete, frozen profiler collection")
    result: dict[str, Any] = {"scope": "derived_profile_diagnostics_not_acceptance", "records": [],
        "mapping": "10 cases in public benchmark order; 1051 tasks/case; discard first 51, retain final 1000",
        "counter_scope": "raw PMU event counts and reported time/ratios; bypass misses may not allocate; not universal hit percentages"}
    for item in collection["records"]:
        directory = Path(item["profile_directory"])
        files = list(directory.rglob("op_summary*.csv"))
        if len(files) != 1:
            raise ValueError("Expected exactly one operator summary CSV per collection")
        path = files[0]
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 10510 or {r["Op Name"] for r in rows} != {"T5SplitQkvAppendCache"}:
            raise ValueError("Profiler task count/type mapping does not match the original benchmark")
        rows.sort(key=lambda r: float(r["Task Start Time(us)"]))
        benchmark = _read(directory / "benchmark.json")
        if benchmark["protocol"] != {"warmup": 50, "iterations": 200, "rounds": 5} or len(benchmark["cases"]) != 10:
            raise ValueError("Unexpected benchmark protocol or case count")
        for index, case in enumerate(benchmark["cases"]):
            timed = rows[index * 1051 + 51:(index + 1) * 1051]
            fields = ["Task Duration(us)", "aiv_scalar_time(us)", "aiv_scalar_ratio",
                      "aiv_mte2_time(us)", "aiv_mte2_ratio", "aiv_mte3_time(us)", "aiv_mte3_ratio",
                      "aiv_write_cache_hit", "aiv_write_cache_miss_allocate", "aiv_r0_read_cache_hit",
                      "aiv_r0_read_cache_miss_allocate", "aiv_r1_read_cache_hit", "aiv_r1_read_cache_miss_allocate"]
            result["records"].append({"case_name": case["case_name"], "allocator": item["allocator"],
                "metric": item["metric"], "timed_tasks": len(timed), "block_counts": sorted({r["Block Num"] for r in timed}),
                "csv_path": str(path), "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "instrumented_api_wall_us_median": case["results"]["custom"]["wall_us_median"],
                "median_reported_fields": {f: _median(timed, f) for f in fields}})
    _write(args.output, result)


if __name__ == "__main__":
    _main()
