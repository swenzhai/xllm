# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Condition allocator pool before the public benchmark: diagnosis, NOT acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primer-bytes", type=int, choices=[0, 1073741312], required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.identity.exists():
        parser.error("Preserve prior evidence")
    import torch
    import torch_npu

    torch.npu.set_device(0)
    if torch.npu.get_device_name(0) != "Ascend910_9382" or torch.npu.device_count() != 1:
        raise ValueError("Require exactly one visible A3 chip")
    before = {"allocated": torch.npu.memory_allocated(), "reserved": torch.npu.memory_reserved()}
    if args.primer_bytes:
        # Deliberate external conditioning. Never insert this into the operator
        # or describe this run as default-allocator acceptance or zero-workspace
        # memory qualification. No values are read from this temporary tensor.
        temporary = torch.empty(args.primer_bytes, dtype=torch.uint8, device="npu:0")
        del temporary
        torch.npu.synchronize()
    conditioned = {"allocated": torch.npu.memory_allocated(), "reserved": torch.npu.memory_reserved(),
                   "segments": torch.npu.memory_snapshot(), "peak_allocated": torch.npu.max_memory_allocated()}
    if conditioned["allocated"] != before["allocated"]:
        raise ValueError("Primer tensor is still active")
    sys.path.insert(0, str(args.benchmark.parent))
    sys.argv = [str(args.benchmark), *args.args[1:]]
    code = 0
    try:
        runpy.run_path(str(args.benchmark), run_name="__main__")
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
    finally:
        torch.npu.synchronize()
        maps = Path("/proc/self/maps").read_text()
        paths = sorted({line.split()[-1] for line in maps.splitlines()
                        if line.split()[-1].startswith("/") and ".so" in line.split()[-1]
                        and ("t5qkv" in line.lower() or "libcust_" in line or "libtorch_npu" in line)})
        smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        result = {"scope": "externally_conditioned_pool_diagnostic_NOT_ACCEPTANCE",
            "pid": os.getpid(), "device_name": torch.npu.get_device_name(0),
            "visible_devices": torch.npu.device_count(), "primer_bytes": args.primer_bytes,
            "allocator_env": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
            "visible_device_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "before": before, "after_priming": conditioned,
            "after_benchmark_segments": torch.npu.memory_snapshot(),
            "loaded_files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            "npu_smi_stdout": smi.stdout, "npu_smi_stderr": smi.stderr,
            "original_exit_code": code, "torch": torch.__version__, "torch_npu": torch_npu.__version__}
        args.identity.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(_main())
