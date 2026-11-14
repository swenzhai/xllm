# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""External pool-layout intervention for cause isolation, never an acceptance run."""

import argparse
from pathlib import Path
import sys

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
    before = _snapshot(files)
    reports.mkdir(parents=True, exist_ok=False)
    _write(reports / "pretest_manifest.json", before)
    result = {"scope": "external_pool_conditioning_diagnostic_NOT_ACCEPTANCE",
              "status": "running", "records": [], "full_acceptance": False}
    _write(reports / "summary.json", result)
    try:
        for batch in range(3):
            devices = config["visible_devices"] if batch % 2 == 0 else list(reversed(config["visible_devices"]))
            for device in devices:
                primers = [0, 1073741312] if (batch + int(device)) % 2 == 0 else [1073741312, 0]
                for primer in primers:
                    if _snapshot(files) != before:
                        raise ValueError("Identity changed before diagnostic")
                    item = reports / f"batch{batch}_chip{device}_primer{primer}"
                    item.mkdir()
                    command = [sys.executable, str(HERE / "pool_probe.py"), "--primer-bytes", str(primer),
                        "--benchmark", str(package / "benchmark.py"), "--identity", str(item / "identity.json"),
                        "--", "--solution-file", str(Path(config["new"]["project"]) / "solution.py"),
                        "--device", "0", "--report", str(item / "benchmark.json")]
                    rc = _child(command, package, _environment(config, "new", "default", device), item / "benchmark.log")
                    _identity(item / "identity.json", config, "new")
                    checked = _evaluate(_read(item / "benchmark.json"), _read(package / "performance_targets.json"))
                    if rc != (0 if checked["status"] == "passed" else 1):
                        raise ValueError("Original benchmark exit disagrees with checker")
                    result["records"].append({"batch": batch, "visible_device": device,
                        "primer_bytes": primer, "directory": str(item), "instrumented_conditioned_checks": checked})
                    _write(reports / "summary.json", result)
        result["status"] = "completed_diagnostic_NOT_ACCEPTANCE"
    except (ValueError, OSError, KeyError, TypeError, KeyboardInterrupt) as error:
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
