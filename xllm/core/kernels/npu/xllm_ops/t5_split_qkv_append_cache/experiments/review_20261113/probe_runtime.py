# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Collect runtime identity; do not launch the operator or a timing workload."""

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys

from evidence import _write


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Choose a new identity report")
    import torch
    import torch_npu

    torch.npu.set_device(0)
    device = torch.npu.get_device_name(0)
    smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
    _write(args.report, {"scope": "runtime_identity_only", "npu_executed": False,
                        "device_name": device, "visible_devices": torch.npu.device_count(),
                        "python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                        "os_release": Path("/etc/os-release").read_text(),
                        "allocator": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
                        "npu_smi_exit_code": smi.returncode, "npu_smi_stdout": smi.stdout,
                        "npu_smi_stderr": smi.stderr})
    return 0 if device == "Ascend910_9382" and torch.npu.device_count() == 1 else 2


if __name__ == "__main__":
    raise SystemExit(_main())
