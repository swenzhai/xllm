# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Compile actual candidate helpers and run existing CPU mapping checks."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from evidence import _candidate, _check, _child, _read, _write

HERE = Path(__file__).resolve().parent


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cxx", default="c++")
    args = parser.parse_args()
    root, reports = args.prepared_root.resolve(), args.report_dir.resolve()
    cxx = shutil.which(args.cxx)
    if cxx is None:
        parser.error("A host C++17 compiler is required")
    reports.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "scope": "host_mapping_and_policy_only",
              "npu_executed": False, "candidates": []}
    _write(reports / "summary.json", result)
    for entry in _read(root / "prepared_manifest.json")["candidates"]:
        name = entry["name"]
        _candidate(root, name)
        project = root / name / "project"
        binary = reports / (name + "_policy_check")
        commands = [
            ([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O2", "-I", str(project),
              str(HERE / "check_candidate_policy.cpp"), "-o", str(binary)], "compile"),
            ([str(binary)], "policy"),
            ([sys.executable, str(project / "tests/check_task_mapping.py"), "--cxx", cxx], "mapping"),
        ]
        for command, label in commands:
            rc = _child(command, project, dict(os.environ), reports / (name + "_" + label + ".log"))
            if rc != 0:
                result.update(status="failed", failed_candidate=name, failed_command=label)
                _write(reports / "summary.json", result)
                return 1
        policy = _read(reports / (name + "_policy.log"))
        mapping = _read(reports / (name + "_mapping.log"))
        expected_bytes = (67108864 + entry["normal_value_rows"] * 262144
                          + entry["normal_key_rows"] * 262272 + 131072)
        if (policy["value_rows"] != entry["normal_value_rows"]
                or policy["normal_key_rows"] != entry["normal_key_rows"]
                or policy["read_balance_enabled"] != (entry["policy"] == "read_balance")
                or policy["normal_bytes"] != expected_bytes
                or policy["budget_bytes"] != entry["budget_mib"] * 1024 * 1024
                or policy["tiling_bytes"] != 52 or mapping["status"] != "passed"
                or mapping["shape_core_offset_checks"] != 104 or mapping["decoded_tasks"] != 171584):
            raise ValueError("Actual compiled policy/mapping differs from candidate contract")
        _check(project, entry["source_files"])
        result["candidates"].append({"name": name, "policy": policy, "mapping": mapping})
    result["status"] = "passed"
    _write(reports / "summary.json", result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
