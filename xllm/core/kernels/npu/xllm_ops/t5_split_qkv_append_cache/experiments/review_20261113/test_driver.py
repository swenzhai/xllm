# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Host-only driver failure-path checks. Mock children are never NPU evidence."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from typing import Any

from evidence import _read, _write
from run_matrix import _evaluate, _execute, _schedule


def _targets() -> dict[str, Any]:
    return {"cases": [{"case_name": f"synthetic_{n}", "max_wall_us": 75} for n in range(10)]}


def _report(slow: bool = False) -> dict[str, Any]:
    cases, checks = [], []
    for index in range(10):
        rounds = [76, 77, 78, 79, 80] if slow and index == 0 else [10, 11, 12, 13, 14]
        cases.append({"case_name": f"synthetic_{index}",
                      "accuracy": [{"output": n, "bit_mismatches": 0}
                                   for n in ["query", "key", "value"]],
                      "results": {"custom": {"wall_us_rounds": rounds, "wall_us_median": rounds[2],
                          "device_event_us_rounds": [8, 9, 10, 11, 12], "device_event_us_median": 10}}})
        checks.append({"case_name": f"synthetic_{index}", "status": "failed" if slow and index == 0 else "passed",
                       "wall_us_rounds": rounds, "wall_us_median": rounds[2], "max_wall_us": 75})
    status = "failed" if slow else "passed"
    return {"status": status, "npu_executed": True, "protocol": {"warmup": 50, "iterations": 200, "rounds": 5},
            "cases": cases, "performance_checks": {"status": status, "checks": checks}}


class DriverTest(unittest.TestCase):
    def test_schedule_covers_both_contexts_without_favoring_fixed_order(self) -> None:
        plan = _schedule(["control", "c1", "c2"], 3)
        self.assertEqual(len(plan), 18)
        self.assertEqual(len({(p["batch"], p["candidate"], p["allocator"]) for p in plan}), 18)
        self.assertEqual([p["candidate"] for p in plan[:3]], ["control", "c1", "c2"])
        self.assertEqual([p["candidate"] for p in plan[6:9]], ["c1", "c2", "control"])
        self.assertEqual(plan[0]["allocator"], "default")
        self.assertEqual(plan[6]["allocator"], "page1g")

    def test_original_failures_are_retained_and_bad_evidence_is_rejected(self) -> None:
        report = _report(True)
        saved = copy.deepcopy(report)
        self.assertEqual(_evaluate(report, _targets())["status"], "failed")
        self.assertEqual(saved, report)
        for change in ["median", "missing", "duplicate", "protocol", "accuracy", "nonfinite", "checker"]:
            value = _report()
            with self.subTest(change=change):
                if change == "median":
                    value["cases"][0]["results"]["custom"]["wall_us_median"] = 1
                elif change == "missing":
                    value["cases"].pop()
                elif change == "duplicate":
                    value["cases"][1] = value["cases"][0]
                elif change == "protocol":
                    value["protocol"]["iterations"] = 1
                elif change == "accuracy":
                    value["cases"][0]["accuracy"][0]["bit_mismatches"] = 1
                elif change == "nonfinite":
                    value["cases"][0]["results"]["custom"]["wall_us_rounds"][0] = float("nan")
                else:
                    value["performance_checks"]["checks"][0]["status"] = "failed"
                with self.assertRaises(ValueError):
                    _evaluate(value, _targets())

    def _exercise(self, scenario: str) -> tuple[int, dict[str, Any], int]:
        with tempfile.TemporaryDirectory(prefix="t5qkv-driver-mock-") as temporary:
            root = Path(temporary)
            project = root / "control/project"
            project.mkdir(parents=True)
            (project / "source.h").write_text("// synthetic source, not an operator\n")
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            patch_file = project.parent / "candidate.patch"
            patch_file.write_text("")
            entry = {"name": "control", "source_files": {"source.h": digest(project / "source.h")},
                     "patch_sha256": digest(patch_file)}
            _write(project.parent / "source_manifest.json", entry)
            artifact = project.parent / "synthetic.so"
            artifact.write_bytes(b"not an executable library")
            _write(project.parent / "build_manifest.json", {"status": "built_a3_unvalidated",
                   "source_files": entry["source_files"], "artifacts": {artifact.name: digest(artifact)}})
            package = root / "official_task_package"
            package.mkdir()
            _write(package / "performance_targets.json", _targets())
            _write(root / "prepared_manifest.json", {"candidates": [entry],
                   "official_files": {"performance_targets.json": digest(package / "performance_targets.json")}})
            reports = root / "reports"
            reports.mkdir()
            calls = []

            def child(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> int:
                log.write_bytes(b"MOCK CHILD OUTPUT: not hardware evidence\n")
                output = Path(command[command.index("--report") + 1])
                script = Path(command[1]).name
                if script == "probe_runtime.py":
                    _write(output, {"device_name": "910B4" if scenario == "wrong_chip" else "Ascend910_9382",
                                    "visible_devices": 1})
                    return 2 if scenario == "wrong_chip" else 0
                if script == "verify.py":
                    _write(output, {"status": "passed", "npu_executed": True, "cases": [{}] * 35})
                    return 0
                custom = "taskbook_baseline.py" not in command[command.index("--solution-file") + 1]
                if custom:
                    calls.append(command)
                slow = scenario == "threshold_failure" and custom and len(calls) == 1
                _write(output, _report(slow))
                if scenario == "changed_artifact" and custom:
                    artifact.write_bytes(b"mutated while the benchmark ran")
                return 1 if slow else 0

            plan = [{"batch": n, "allocator": "default", "candidate": "control"} for n in range(2)]
            with patch("run_matrix._child", side_effect=child):
                rc = _execute(root, package, reports, ["control"], plan, "unused-python", "screen")
            return rc, _read(reports / "summary.json"), len(calls)

    def test_failed_batch_does_not_hide_or_cancel_later_batches(self) -> None:
        rc, summary, calls = self._exercise("threshold_failure")
        self.assertEqual((rc, calls), (1, 2))
        self.assertEqual([r["status"] for r in summary["records"]], ["failed", "passed"])
        self.assertTrue(summary["source_and_artifacts_unchanged"])
        self.assertFalse(summary["full_acceptance"])

    def test_wrong_chip_blocks_every_benchmark(self) -> None:
        rc, summary, calls = self._exercise("wrong_chip")
        self.assertEqual((rc, calls), (1, 0))
        self.assertFalse(summary["npu_executed"])

    def test_changed_artifact_invalidates_successful_samples(self) -> None:
        rc, summary, calls = self._exercise("changed_artifact")
        self.assertEqual((rc, calls), (1, 1))
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["source_and_artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main()
