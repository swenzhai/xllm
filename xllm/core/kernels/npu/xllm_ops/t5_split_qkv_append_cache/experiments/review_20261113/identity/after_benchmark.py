# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Run the unmodified public script, then collect loaded-library/device identity."""

import argparse
import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence import _write


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--expected-prefix", required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.identity.exists():
        parser.error("Identity report already exists")
    sys.path.insert(0, str(args.script.parent))
    sys.argv = [str(args.script), *args.args[1:]]
    code = 0
    try:
        runpy.run_path(str(args.script), run_name="__main__")
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
    finally:
        # No imports, allocations or measurement hooks run before/inside the
        # original benchmark. All identity work occurs after its final result.
        import torch
        import torch_npu

        torch.npu.synchronize()
        maps = Path("/proc/self/maps").read_text()
        paths = sorted({line.split()[-1] for line in maps.splitlines()
                        if line.split()[-1].startswith("/") and ".so" in line.split()[-1]
                        and ("t5qkv" in line.lower() or "libcust_" in line or "libtorch_npu" in line)})
        smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        _write(args.identity, {"scope": "post_benchmark_identity_no_timing_hooks",
            "pid": os.getpid(), "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "device_name": torch.npu.get_device_name(0), "visible_devices": torch.npu.device_count(),
            "requested_visible_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "allocator": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
            "expected_prefix": args.expected_prefix,
            "loaded_files": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            "allocator_snapshot_after": torch.npu.memory_snapshot(), "maps": maps,
            "npu_smi_exit_code": smi.returncode, "npu_smi_stdout": smi.stdout,
            "npu_smi_stderr": smi.stderr, "original_exit_code": code})
    return code


if __name__ == "__main__":
    raise SystemExit(_main())
