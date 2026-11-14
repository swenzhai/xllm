# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Instrument allocator history around the unchanged benchmark; not acceptance."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import site
import subprocess
import sys


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.trace.exists() or args.identity.exists():
        parser.error("Preserve existing evidence")
    site.addsitedir(args.site_packages)
    import torch
    import torch_npu

    torch.npu.set_device(0)
    if torch.npu.get_device_name(0) != "Ascend910_9382" or torch.npu.device_count() != 1:
        raise ValueError("Require one visible target A3 chip")
    # The installed torch_npu implementation was checked: context=None records
    # alloc/free events without stack collection. Instrumented timings are never
    # used as acceptance samples, even though the public script is unchanged.
    torch.npu.memory._record_memory_history(enabled="all", context=None, max_entries=200000)
    sys.path.insert(0, str(args.benchmark.parent))
    sys.argv = [str(args.benchmark), *args.args[1:]]
    code = 0
    try:
        runpy.run_path(str(args.benchmark), run_name="__main__")
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
    finally:
        torch.npu.synchronize()
        snapshot = torch.npu.memory._snapshot()
        torch.npu.memory._record_memory_history(enabled=None)
        with gzip.open(args.trace, "xt", encoding="utf-8") as stream:
            json.dump(snapshot, stream)
        maps = Path("/proc/self/maps").read_text()
        paths = sorted({line.split()[-1] for line in maps.splitlines()
                        if line.split()[-1].startswith("/") and ".so" in line.split()[-1]
                        and ("t5qkv" in line.lower() or "libcust_" in line or "libtorch_npu" in line)})
        smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        implementation = Path(torch_npu.npu.memory.__file__)
        identity = {"scope": "instrumented_memory_history_diagnostic_not_acceptance",
            "pid": os.getpid(), "device_name": torch.npu.get_device_name(0),
            "visible_devices": torch.npu.device_count(),
            "visible_device_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "allocator_env": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
            "history_options": {"enabled": "all", "context": None, "max_entries": 200000},
            "implementation": str(implementation),
            "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
            "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
            "npu_smi_stdout": smi.stdout, "npu_smi_stderr": smi.stderr,
            "npu_smi_exit_code": smi.returncode,
            "loaded_files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            "original_exit_code": code}
        args.identity.write_text(json.dumps(identity, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(_main())
