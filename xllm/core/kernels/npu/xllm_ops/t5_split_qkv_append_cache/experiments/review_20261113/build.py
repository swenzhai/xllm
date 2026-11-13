# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Build one fresh isolated A3 candidate; compilation is not NPU validation."""

import argparse
import json
from pathlib import Path
import os
import sys

from evidence import _candidate, _check, _child, _environment, _hashes, _write


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    root = args.prepared_root.resolve()
    entry = _candidate(root, args.candidate)
    if not os.environ.get("ASCEND_HOME_PATH"):
        parser.error("Source the target CANN environment and activate Python dependencies first")
    directory = root / args.candidate
    project = directory / "project"
    if any(project.glob("build*")) or (directory / "install").exists():
        parser.error("Use freshly prepared trees; preserve partial/old builds instead of overwriting")
    logs = directory / "build_records"
    logs.mkdir(exist_ok=False)
    env = _environment(root, args.candidate, "default")
    env.update(T5QKV_BUILD_DIR=str(project / "build_sdk"), T5QKV_BUILD_JOBS="2", MAX_JOBS="2")
    commands = [
        ["bash", "scripts/build.sh"],
        ["bash", "scripts/install.sh", str(project / "build_sdk/custom_opp_ubuntu_aarch64.run"),
         str(directory / "install")],
        ["bash", "scripts/build_framework.sh"],
    ]
    record = {"status": "building", "scope": "a3_target_build_only", "npu_executed": False,
              "source_files": entry["source_files"], "python": sys.version}
    _write(directory / "build_manifest.json", record)
    for index, command in enumerate(commands):
        rc = _child(command, project, env, logs / f"{index:02}.log")
        _check(project, entry["source_files"])
        if rc != 0:
            record.update(status="build_failed", failed_command=index, exit_code=rc)
            _write(directory / "build_manifest.json", record)
            return 1
    extensions = list(project.glob("_t5qkv*.so"))
    api = directory / "install/vendors/t5qkv/op_api/lib/libcust_opapi.so"
    direct = project / "build_direct/libt5qkv_direct.so"
    package = project / "build_sdk/custom_opp_ubuntu_aarch64.run"
    if len(extensions) != 1 or not all(p.is_file() for p in [api, direct, package]):
        raise ValueError("Expected exactly one extension, direct library, API library and SDK package")
    artifacts = [p for p in (directory / "install").rglob("*") if p.is_file()]
    artifacts += [extensions[0], direct, package]
    record.update(status="built_a3_unvalidated", artifacts=_hashes(
        directory, sorted({str(p.relative_to(directory)) for p in artifacts})))
    _write(directory / "build_manifest.json", record)
    print(json.dumps({"candidate": args.candidate, "status": record["status"],
                      "npu_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
