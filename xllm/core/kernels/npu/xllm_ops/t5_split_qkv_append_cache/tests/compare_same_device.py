# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Pair original public benchmark runs in independent NPU processes.

Each algorithm uses the unchanged fixture generator, verifier and 50/200/5 timer.
Independent processes prevent a preceding algorithm's temporary allocations from
changing the other's allocator setup. They use identical generated input values,
not shared device addresses. Original mixed-process diagnostics remain separate.
The reference is qualified by all 35 public accuracy cases. Submission absolute
limits apply to the custom implementation; reference threshold failures are
recorded verbatim and are not described as reference performance acceptance.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _child(name: str, command: list[str], cwd: Path, directory: Path) -> int:
    with (directory / (name + ".log")).open("wb") as output:
        result = subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    (directory / (name + "_execution.json")).write_text(json.dumps({
        "command": command, "cwd": str(cwd), "exit_code": result.returncode}, indent=2))
    return result.returncode


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--solution-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    package, solution = args.package_dir.resolve(), args.solution_file.resolve()
    report_path = args.report.resolve()
    directory = report_path.parent / report_path.stem
    directory.mkdir(parents=True, exist_ok=False)
    baseline = Path(__file__).with_name("taskbook_baseline.py").resolve()
    common = ["--device", str(args.device)]
    qualification_rc = _child("baseline_accuracy", [sys.executable, str(package / "verify.py"),
        "--solution-file", str(baseline), "--report", str(directory / "baseline_accuracy.json"), *common], package, directory)
    if qualification_rc != 0:
        raise RuntimeError("Reference accuracy qualification failed; inspect original log")
    exit_codes = {}
    for name, entry in [("custom", solution), ("baseline", baseline)]:
        exit_codes[name] = _child(name, [sys.executable, str(package / "benchmark.py"),
            "--solution-file", str(entry), "--report", str(directory / (name + ".json")), *common], package, directory)
    current, reference = _read(directory / "custom.json"), _read(directory / "baseline.json")
    # Preserve a complete paired diagnostic even when the custom misses its
    # absolute limits. The aggregate and exit status must still report failure.
    assert current.get("npu_executed") and not current.get("error")
    limits = {c["case_name"]: c["max_wall_us"] for c in _read(package / "performance_targets.json")["cases"]}
    reference_cases = {c["case_name"]: c for c in reference["cases"]}
    assert set(reference_cases) == set(limits)
    assert {c["case_name"] for c in current["cases"]} == set(limits)
    assert reference.get("npu_executed") and not reference.get("error")
    assert reference["protocol"] == current["protocol"] == {"warmup": 50, "iterations": 200, "rounds": 5}
    sys.path.insert(0, str(package))
    spec = importlib.util.spec_from_file_location("fixture_generator", package / "generate_data.py")
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    fixtures = {c["case_name"]: c for c in _read(package / "case.json")}
    cases = []
    for case in current["cases"]:
        name = case["case_name"]
        previous = reference_cases[name]
        assert all(c["bit_mismatches"] == 0 for c in previous["accuracy"] + case["accuracy"])
        custom = case["results"]["custom"]
        base = previous["results"]["custom"]
        assert len(custom["wall_us_rounds"]) == len(base["wall_us_rounds"]) == 5
        cases.append({"case_name": name, "accuracy": {"custom": case["accuracy"],
            "views_concat_q_copy": previous["accuracy"]}, "results": {"custom": custom,
            "views_concat_q_copy": base}, "selected_baseline": "views_concat_q_copy",
            "input_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in generator.generate(fixtures[name])],
            "max_wall_us": limits[name],
            "absolute_status": "passed" if custom["wall_us_median"] <= limits[name] else "failed",
            "ratio": custom["wall_us_median"] / base["wall_us_median"]})
    passed = (exit_codes["custom"] == 0 and current["status"] == "passed"
              and all(case["absolute_status"] == "passed" for case in cases))
    result = {"status": "passed" if passed else "failed", "npu_executed": True, "cases": cases,
        "protocol": current["protocol"], "environment": current["environment"],
        "runtime_options": {k: os.environ.get(k) for k in ["ASCEND_RT_VISIBLE_DEVICES", "PYTORCH_NPU_ALLOC_CONF", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING"]},
        "comparison_method": "unchanged public benchmark; independent processes on same device and generated input values",
        "reference_qualification": "35 public accuracy cases; submission absolute thresholds are not reference acceptance criteria",
        "original_child_exit_codes": exit_codes, "original_reference_status": reference["status"],
        "raw_directory": str(directory), "fixture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.iterdir() if p.is_file()},
        "implementation_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [solution, baseline, Path(__file__)]}}
    report_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
