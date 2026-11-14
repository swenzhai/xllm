# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Real direct launch with aligned/unaligned supplied outputs and guard words."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path


class _Tiling(ctypes.Structure):
    _fields_ = [("word" + str(i), ctypes.c_uint32) for i in range(13)]


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-file", type=Path, required=True)
    parser.add_argument("--tiling-words", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve prior results")
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    project = args.solution_file.parent
    plan = json.loads(args.tiling_words.read_text())
    header = project / "common/t5_split_qkv_append_cache_tiling.h"
    if hashlib.sha256(header.read_bytes()).hexdigest() != plan["header_sha256"]:
        raise ValueError("CPU tiling words do not match the built header")
    tiling = _Tiling(*plan["tiling_words"])
    if ctypes.sizeof(tiling) != 52:
        raise ValueError("Unexpected tiling ABI")
    torch.npu.set_device(0)
    acl = ctypes.CDLL("/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so")
    acl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    acl.aclrtCreateStream.restype = ctypes.c_int32
    for name in ["aclrtSynchronizeStream", "aclrtDestroyStream"]:
        function = getattr(acl, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int32
    path = project / "build_direct/libt5qkv_direct.so"
    launch = ctypes.CDLL(str(path)).t5qkv_launch
    launch.argtypes = [ctypes.c_uint32, *([ctypes.c_void_p] * 7), _Tiling]
    launch.restype = ctypes.c_uint32
    stream = ctypes.c_void_p()
    if acl.aclrtCreateStream(ctypes.byref(stream)) != 0:
        raise RuntimeError("Cannot create diagnostic stream")
    report = {"scope": "direct_supplied_output_alignment_and_guards", "status": "running",
        "npu_executed": False, "records": [], "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    try:
        shapes = [(16, 1, 3072), (16, 16, 2048, 64), (16, 16, 2048, 64)]
        arrays = [(np.arange(np.prod(s), dtype=np.uint16) + i * 997).reshape(s)
                  for i, s in enumerate(shapes)]
        for index in (1, 2):
            arrays[index] ^= np.arange(256, dtype=np.uint16).reshape(16, 16, 1, 1) * np.uint16(257)
        qkv = arrays[0].reshape(16, 3, 16, 64)
        expected = [qkv[:, 0, :, None, :].copy(),
            np.concatenate((arrays[1], qkv[:, 1, :, None, :]), axis=2),
            np.concatenate((arrays[2], qkv[:, 2, :, None, :]), axis=2)]
        for dtype in (torch.float16, torch.bfloat16):
            for offset in (0, 1):
                inputs = [torch.from_numpy(a.view(np.int16)).to("npu:0").view(dtype) for a in arrays]
                guards = [torch.full((a.size + 33,), 21930, dtype=torch.int16, device="npu:0") for a in expected]
                offsets = [0, offset, 0]
                outputs = [raw[o:o + a.size].reshape(a.shape).view(dtype)
                           for raw, o, a in zip(guards, offsets, expected)]
                if (outputs[1].data_ptr() % 512 == 0) != (offset == 0):
                    raise AssertionError("Test did not exercise the intended output alignment")
                torch.npu.synchronize()
                rc = launch(48, stream, *[ctypes.c_void_p(t.data_ptr()) for t in (*inputs, *outputs)], tiling)
                if rc != 0 or acl.aclrtSynchronizeStream(stream) != 0:
                    raise RuntimeError("Direct launch or synchronization failed")
                report["npu_executed"] = True
                for output, wanted in zip(outputs, expected):
                    observed = output.view(torch.int16).cpu().numpy().view(np.uint16)
                    if not np.array_equal(observed, wanted):
                        raise AssertionError("Supplied output raw-bit mismatch")
                for tensor, wanted in zip(inputs, arrays):
                    if not np.array_equal(tensor.view(torch.int16).cpu().numpy().view(np.uint16), wanted):
                        raise AssertionError("Input changed")
                for raw, start, wanted in zip(guards, offsets, expected):
                    observed = raw.cpu().numpy()
                    if not np.all(observed[:start] == 21930) or not np.all(observed[start + wanted.size:] == 21930):
                        raise AssertionError("Output guard overwritten")
                report["records"].append({"dtype": str(dtype), "key_output_offset_elements": offset,
                    "key_data_mod512": outputs[1].data_ptr() % 512, "status": "passed",
                    "raw_bits_equal": True, "guard_words_preserved": True})
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                del inputs, outputs, guards, raw, output
        report["status"] = "passed"
    except (ValueError, RuntimeError, AssertionError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        acl.aclrtSynchronizeStream(stream)
        rc = acl.aclrtDestroyStream(stream)
        report["stream_destroy_return_code"] = rc
        if rc != 0:
            report["status"] = "failed"
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
