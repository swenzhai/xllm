# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Shared evidence helpers for isolated offline preparation and later A3 runs."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _hashes(root: Path, names: list[str]) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _check(root: Path, expected: dict[str, str]) -> None:
    actual = _hashes(root, list(expected))
    if actual != expected:
        raise ValueError(f"Source/artifact identity changed under {root}")


def _candidate(root: Path, name: str) -> dict[str, Any]:
    entries = {c["name"]: c for c in _read(root / "prepared_manifest.json")["candidates"]}
    if name not in entries:
        raise ValueError(f"Unknown candidate: {name}")
    if _read(root / name / "source_manifest.json") != entries[name]:
        raise ValueError("Candidate manifests disagree")
    entry = entries[name]
    _check(root / name / "project", entry["source_files"])
    _check(root / name, {"candidate.patch": entry["patch_sha256"]})
    return entry


def _environment(root: Path, name: str, allocator: str) -> dict[str, str]:
    env = dict(os.environ)
    prefix = root / name / "install/vendors/t5qkv"
    env.update(ASCEND_CUSTOM_OPP_PATH=str(prefix),
               T5QKV_OPAPI_LIB=str(prefix / "op_api/lib/libcust_opapi.so"),
               ASCEND_RT_VISIBLE_DEVICES="0")
    env["LD_LIBRARY_PATH"] = ":".join([
        str(prefix / "op_api/lib"), str(root / name / "project/build_direct"),
        env.get("LD_LIBRARY_PATH", "")])
    for key in ["PYTORCH_NPU_ALLOC_CONF", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING"]:
        env.pop(key, None)
    if allocator == "page1g":
        env["PYTORCH_NPU_ALLOC_CONF"] = "page_size:1g"
    elif allocator != "default":
        raise ValueError(f"Unknown allocator: {allocator}")
    return env


def _child(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> int:
    start = time.time()
    record: dict[str, Any] = {"command": command, "cwd": str(cwd),
                              "started_at_unix": start, "status": "running",
                              "environment": {k: env.get(k) for k in [
                                  "ASCEND_CUSTOM_OPP_PATH", "T5QKV_OPAPI_LIB",
                                  "ASCEND_RT_VISIBLE_DEVICES", "PYTORCH_NPU_ALLOC_CONF",
                                  "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING",
                                  "ASCEND_HOME_PATH", "LD_LIBRARY_PATH"]}}
    path = log.with_suffix(".execution.json")
    _write(path, record)
    with log.open("xb") as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, status="exited", elapsed_seconds=time.time() - start)
    _write(path, record)
    return result.returncode
