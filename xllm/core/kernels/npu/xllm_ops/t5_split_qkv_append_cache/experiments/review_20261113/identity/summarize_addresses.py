# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Reconcile every allocation with the full public call sequence and same-run PMU."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _physical(identity: dict[str, Any]) -> dict[str, Any]:
    chips, process, npu = {}, [], None
    for line in identity["npu_smi_stdout"].splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells:
            continue
        first = cells[0].split()
        if len(first) == 2 and first[0].isdigit() and first[1] == "Ascend910":
            npu = int(first[0])
        if len(first) != 2 or not all(p.isdigit() for p in first):
            continue
        if cells[1].startswith("0000:"):
            chips[(npu, int(first[0]))] = {"npu_id": npu, "chip_id": int(first[0]),
                "physical_id": int(first[1]), "pcie_bus": cells[1]}
        elif len(cells) == 5 and str(identity["pid"]) in [cells[1], cells[-1]]:
            process.append((int(first[0]), int(first[1])))
    if len(process) != 1 or process[0] not in chips:
        raise ValueError("Cannot resolve exactly one live PID to its physical chip")
    return chips[process[0]]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior evidence")
    collection = _read(args.profile_root / "summary.json")
    if collection["status"] != "collected_not_acceptance" or not collection["source_and_artifacts_unchanged"]:
        raise ValueError("Require complete frozen diagnostic collection")
    cases = [c for c in _read(args.package / "case.json") if c.get("performance") is True]
    expected, spans = [], []
    for case in cases:
        inputs = [math.prod(d["shape"]) * 2 for d in case["input_desc"]]
        inputs = [size for size in inputs if size != 0]
        outputs = [math.prod(d["shape"]) * 2 for d in case["output_desc"]]
        begin = len(expected)
        expected += inputs + outputs * 1051
        spans.append((begin, len(inputs), len(expected)))
    result = {"scope": "same_process_virtual_allocation_and_L2_diagnostics_not_acceptance",
              "address_limit": "Virtual addresses/allocator segments; physical HBM page addresses unavailable",
              "allocation_mapping": "Exact complete size sequence: nonempty inputs, then 1051 fresh Q/K/V triples per public case",
              "records": []}
    for job in collection["records"]:
        directory = args.profile_root / Path(job["directory"]).name
        identity = _read(directory / "identity.json")
        path = directory / "memory_trace.json.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity["trace_sha256"]:
            raise ValueError("Trace identity changed")
        with gzip.open(path, "rt") as stream:
            trace = json.load(stream)
        events = [e for seq in trace["device_traces"] for e in seq]
        if len(events) >= 200000:
            raise ValueError("History may have wrapped")
        allocations = [e for e in events if e["action"] == "alloc"]
        if [e["size"] for e in allocations] != expected:
            raise ValueError("Full allocation sequence does not match public inputs and calls")
        csv_files = list(directory.rglob("op_summary*.csv"))
        if len(csv_files) != 1:
            raise ValueError("Expected one operator summary")
        with csv_files[0].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 10510 or {r["Op Name"] for r in rows} != {"T5SplitQkvAppendCache"}:
            raise ValueError("Profiler task sequence does not match the public workload")
        rows.sort(key=lambda r: float(r["Task Start Time(us)"]))
        benchmark = _read(directory / "benchmark.json")
        if benchmark["protocol"] != {"warmup": 50, "iterations": 200, "rounds": 5}:
            raise ValueError("Unexpected timer protocol")
        item = {"visible_device": job["visible_device"], "allocator": job["allocator"],
            "physical_device": _physical(identity), "allocations": len(allocations), "events": len(events),
            "segments": [{k: s.get(k) for k in ["address", "total_size", "segment_type", "is_expandable"]}
                         for s in trace["segments"]], "cases": []}
        for index, (case, span) in enumerate(zip(cases, spans)):
            begin, input_count, end = span
            if benchmark["cases"][index]["case_name"] != case["case_name"]:
                raise ValueError("Public case order differs")
            tensors = allocations[begin + input_count:end]
            triples = [tuple(e["addr"] for e in tensors[i:i + 3]) for i in range(51 * 3, 1051 * 3, 3)]
            counts = Counter(triples)
            timed = rows[index * 1051 + 51:(index + 1) * 1051]
            fields = ["Task Duration(us)", "aiv_r0_read_cache_hit", "aiv_r0_read_cache_miss_allocate",
                      "aiv_r1_read_cache_hit", "aiv_r1_read_cache_miss_allocate",
                      "aiv_write_cache_hit", "aiv_write_cache_miss_allocate"]
            medians = {k: statistics.median(float(r[k]) for r in timed) for k in fields}
            item["cases"].append({"case_name": case["case_name"], "input_addresses":
                [a["addr"] for a in allocations[begin:begin + input_count]],
                "unique_timed_output_triples": len(counts), "timed_output_triples":
                [{"addresses": list(t), "count": n} for t, n in counts.most_common(8)],
                "omitted_rare_triples": max(0, len(counts) - 8), "same_process_L2_medians": medians,
                "instrumented_wall_us_not_acceptance": benchmark["cases"][index]["results"]["custom"]["wall_us_median"]})
        result["records"].append(item)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    _main()
