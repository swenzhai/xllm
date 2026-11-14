# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Read allocation page attributes in isolation; no operator or timing claims."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess


class _Location(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_int32)]


class _Attributes(ctypes.Structure):
    _fields_ = [("location", _Location), ("page_size", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 4)]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve prior evidence")
    import torch
    import torch_npu

    torch.npu.set_device(0)
    if torch.npu.get_device_name(0) != "Ascend910_9382" or torch.npu.device_count() != 1:
        raise ValueError("Require one visible A3 chip")
    header = Path("/usr/local/Ascend/cann-9.1.0/include/acl/acl_rt.h")
    library = Path("/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so").resolve()
    api = ctypes.CDLL(str(library)).aclrtPointerGetAttributes
    api.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Attributes)]
    api.restype = ctypes.c_int32
    if ctypes.sizeof(_Location) != 8 or ctypes.sizeof(_Attributes) != 28:
        raise ValueError("Unexpected ABI layout")
    # Shapes match the large public geometry but this isolated allocator state
    # deliberately does not claim the preceding public-case allocation history.
    shapes = [(16, 1, 3072), (16, 16, 2048, 64), (16, 16, 2048, 64),
              (16, 16, 1, 64), (16, 16, 2049, 64), (16, 16, 2049, 64)]
    tensors = [torch.empty(s, dtype=torch.float16, device="npu:0") for s in shapes]
    records = []
    for role, tensor in zip(["qkv", "pastKey", "pastValue", "query", "key", "value"], tensors):
        attributes = _Attributes()
        rc = api(ctypes.c_void_p(tensor.data_ptr()), ctypes.byref(attributes))
        records.append({"role": role, "virtual_address": tensor.data_ptr(),
            "bytes": tensor.numel() * tensor.element_size(), "api_return_code": rc,
            "page_size_raw": attributes.page_size if rc == 0 else None,
            "location_id_raw": attributes.location.id if rc == 0 else None,
            "location_type_raw": attributes.location.type if rc == 0 else None})
    smi = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
    data = {"scope": "isolated_allocation_page_attributes_no_operator_or_timing",
        "npu_executed": False, "device_memory_allocated": True, "pid": os.getpid(),
        "visible_device_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "allocator_env": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
        "header": str(header), "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "library": str(library), "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "records": records, "allocator_snapshot": torch.npu.memory_snapshot(),
        "npu_smi_stdout": smi.stdout, "npu_smi_exit_code": smi.returncode,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__}
    args.report.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    _main()
