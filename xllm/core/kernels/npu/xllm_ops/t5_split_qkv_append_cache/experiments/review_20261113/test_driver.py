# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Host-only driver failure-path checks. Mock children are never NPU evidence."""

import copy
import hashlib
from pathlib import Path
import tempfile
import subprocess
import sys
import math
import textwrap
import unittest
from unittest.mock import patch
from typing import Any

from evidence import _read, _write
from run_matrix import _evaluate, _execute, _schedule
from run_functional import _boundary_source, _supplied_args


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
    def test_supplied_output_commands_parse_actual_cli(self) -> None:
        from identity.check_supplied_outputs import _parse_args
        root = Path(__file__).resolve().parent
        entries = _read(root / "candidates.json")["candidates"]
        for entry in entries:
            if entry.get("kernel_policy") not in {"distributed_key_output", "flat_output_tiles", "four_plane", "dispersed_key_output", "paired_history", "interleaved_history"}:
                continue
            with self.subTest(candidate=entry["name"]):
                parsed = _parse_args(["--solution-file", "synthetic_solution.py",
                                      "--report", "not_executed.json", *_supplied_args(entry)])
                self.assertEqual(len(_read(parsed.tiling_words)["tiling_words"]), 13)
                self.assertEqual(4 * len(parsed.value_offsets), entry.get("expected_supplied_outputs", 4))

    def test_loaded_library_guard_requires_every_actual_hash(self) -> None:
        from identity.run_crossover import _identity
        with tempfile.TemporaryDirectory(prefix="t5qkv-library-guard-") as temporary:
            root = Path(temporary)
            project, prefix = root / "project", root / "install"
            files = [project / "_t5qkv.test.so", project / "build_direct/libt5qkv_direct.so",
                prefix / "op_api/lib/libcust_opapi.so",
                prefix / "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
                prefix / "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic host-only guard fixture")
            mapped = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            identity = {"device_name": "Ascend910_9382", "visible_devices": 1, "loaded_files": mapped}
            config = {"new": {"project": str(project), "prefix": str(prefix)}}
            path = root / "identity.json"
            _write(path, identity)
            _identity(path, config, "new")
            for missing in files:
                bad = copy.deepcopy(identity)
                del bad["loaded_files"][str(missing)]
                _write(path, bad)
                with self.assertRaises(ValueError):
                    _identity(path, config, "new")
            _write(path, identity)
            files[0].write_bytes(b"changed file")
            with self.assertRaises(ValueError):
                _identity(path, config, "new")

    def test_boundary_metadata_covers_partial_and_zero_admission(self) -> None:
        # Execute only the generated pure-integer metadata prefix, never imports
        # or NPU operations. The C++ policy checker independently checks geometry.
        original = (Path(__file__).resolve().parents[2] / "tests/cache_boundary.py").read_text()
        for budget, expected in [(186, 432), (190, 456)]:
            for zero in [False, True]:
                with self.subTest(budget=budget, zero=zero):
                    source = _boundary_source(original, {"budget_mib": budget, "policy": "input_priority"}, zero)
                    prefix = "    batch, heads, length, dim =" + source.split(
                        "batch, heads, length, dim =", 1)[1].split("    shapes =", 1)[0]
                    namespace: dict[str, Any] = {"math": math}
                    exec(textwrap.dedent(prefix), namespace)
                    self.assertEqual(namespace["cache_rows"], 0 if zero else expected)
                    self.assertEqual(namespace["cache_rows"] * namespace["row_bytes"] % 512, 0)
                    self.assertIn("normal_key_rows=", source)

    def test_profiler_bootstrap_loads_selected_site_and_preserves_exit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="t5qkv-profile-bootstrap-") as temporary:
            root = Path(temporary)
            dependency = root / "site"
            dependency.mkdir()
            (dependency / "bootstrap_dependency.py").write_text("VALUE = 73\n")
            application = root / "application.py"
            application.write_text("import sys, bootstrap_dependency\n"
                                   "assert bootstrap_dependency.VALUE == 73\n"
                                   "assert sys.argv[1:] == ['--original-argument', 'kept']\n"
                                   "raise SystemExit(7)\n")
            identity = root / "identity.json"
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("profile_application.py")),
                "--site-packages", str(dependency), "--identity", str(identity),
                "--benchmark", str(application), "--", "--original-argument", "kept"], capture_output=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(_read(identity)["argv"], [str(application.resolve()), "--original-argument", "kept"])

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
