# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Full raw-bit checks for shared forward/reverse/equal history source addresses."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-file", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Preserve prior evidence")
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    if torch.npu.get_device_name(0) != "Ascend910_9382" or torch.npu.device_count() != 1:
        raise RuntimeError("Require one visible A3 chip")
    spec = importlib.util.spec_from_file_location("submitted_layout_solution", args.solution_file)
    if spec is None or spec.loader is None:
        raise ImportError(str(args.solution_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = {"scope": "additional_source_layout_regression_not_official_case_count", "status": "running",
        "npu_executed": False, "records": [], "test_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    elements, row_elements = 256 * 2048 * 64, 2048 * 64
    bits = np.arange(2 * elements + 1, dtype=np.uint16)
    bits[:-1].reshape(512, row_elements)[:] ^= (np.arange(512, dtype=np.uint16)[:, None] * np.uint16(257))
    bits[-1] ^= np.uint16((512 * 257) & 65535)
    qkv_bits = (np.arange(16 * 3 * 16 * 64, dtype=np.uint16) + np.uint16(997)).reshape(16, 1, 3072)
    qkv = qkv_bits.reshape(16, 3, 16, 64)
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for layout, key_start, value_start in [("forward_gap2B", 0, elements + 1),
                    ("reverse_gap2B", elements + 1, 0), ("equal_sources", 0, 0)]:
                base = torch.from_numpy(bits.view(np.int16)).to("npu:0")
                qkv_device = torch.from_numpy(qkv_bits.view(np.int16)).to("npu:0").view(dtype)
                key = base[key_start:key_start + elements].reshape(16, 16, 2048, 64).view(dtype)
                value = base[value_start:value_start + elements].reshape(16, 16, 2048, 64).view(dtype)
                expected = [qkv[:, 0, :, None, :].copy(),
                    np.concatenate((bits[key_start:key_start + elements].reshape(16, 16, 2048, 64), qkv[:, 1, :, None, :]), axis=2),
                    np.concatenate((bits[value_start:value_start + elements].reshape(16, 16, 2048, 64), qkv[:, 2, :, None, :]), axis=2)]
                gap = value.data_ptr() - key.data_ptr()
                if gap != 2 * (value_start - key_start):
                    raise AssertionError("Controlled input view addresses do not match")
                for entry in ("run", "run_direct"):
                    record = {"entry": entry, "dtype": str(dtype), "layout": layout,
                        "source_gap_bytes": gap, "key_source": key.data_ptr(), "value_source": value.data_ptr(), "status": "running"}
                    report["records"].append(record)
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
                    outputs = getattr(module, entry)([qkv_device, key, value], {"headNum": 16, "headDim": 64})
                    torch.npu.synchronize()
                    report["npu_executed"] = True
                    storage = [t.untyped_storage().data_ptr() for t in outputs]
                    input_storage = {t.untyped_storage().data_ptr() for t in (qkv_device, key, value)}
                    if len(outputs) != 3 or len(set(storage)) != 3 or set(storage) & input_storage:
                        raise AssertionError("Outputs are not independent")
                    for tensor, wanted in zip(outputs, expected):
                        if not np.array_equal(tensor.view(torch.int16).cpu().numpy().view(np.uint16), wanted):
                            raise AssertionError("Raw output bits differ")
                    if not np.array_equal(base.cpu().numpy().view(np.uint16), bits):
                        raise AssertionError("Shared input storage or gap changed")
                    if not np.array_equal(qkv_device.view(torch.int16).cpu().numpy().view(np.uint16), qkv_bits):
                        raise AssertionError("QKV input changed")
                    record.update(status="passed", raw_bits_equal=True, inputs_unchanged=True,
                        output_mod512=[t.data_ptr() % 512 for t in outputs], independent_outputs=True)
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
                    del outputs, tensor
                del base, qkv_device, key, value, expected
        report["status"] = "passed"
    except (ValueError, RuntimeError, AssertionError, TypeError, OSError) as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
