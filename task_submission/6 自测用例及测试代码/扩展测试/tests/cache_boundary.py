# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Check raw bits across a large, unaligned cache-policy row boundary."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "npu_executed": False, "records": [],
        "test_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    import numpy as np
    import torch
    import torch_npu  # noqa: F401: registers the backend.

    spec = importlib.util.spec_from_file_location("submitted_solution", args.solution_file)
    if spec is None or spec.loader is None:
        raise ImportError(str(args.solution_file))
    solution = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = solution
    spec.loader.exec_module(solution)
    batch, heads, length, dim = 4, 128, 1024, 65
    query_elements = batch * heads * dim
    row_bytes = 2 * (length + 1) * dim
    cache_rows = max(0, 176 * 1024**2 - 4 * query_elements * (length + 2)) // row_bytes
    if not 0 < cache_rows < batch * heads or cache_rows * row_bytes % 32 == 0:
        raise AssertionError("case must cross the cache policy inside an unaligned row boundary")
    shapes = [(batch, 1, 3 * heads * dim), (batch, heads, length, dim)]
    shapes.append(shapes[1])
    arrays = [(np.arange(np.prod(shape), dtype=np.uint16) + index * 997).reshape(shape)
              for index, shape in enumerate(shapes)]
    qkv = arrays[0].reshape(batch, 3, heads, dim)
    expected = [qkv[:, 0, :, None, :].copy()]
    for index in (1, 2):
        expected.append(np.concatenate((arrays[index], qkv[:, index, :, None, :]), axis=2))
    torch.npu.set_device(0)
    for entry in ("run", "run_direct"):
        for dtype in (torch.float16, torch.bfloat16):
            for offset in (0, 1):
                item = {"entry": entry, "dtype": str(dtype), "storage_offset": offset,
                        "status": "running"}
                report["records"].append(item)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                try:
                    inputs = []
                    for array in arrays:
                        raw = torch.empty(array.size + offset, dtype=torch.int16, device="npu:0")
                        view = raw[offset:].reshape(array.shape)
                        view.copy_(torch.from_numpy(array.view(np.int16)))
                        inputs.append(view.view(dtype))
                        if inputs[-1].storage_offset() != offset:
                            raise AssertionError("unexpected input offset")
                    outputs = getattr(solution, entry)(inputs, {"headNum": heads, "headDim": dim})
                    torch.npu.synchronize()
                    report["npu_executed"] = True
                    pointers = [t.untyped_storage().data_ptr() for t in (*inputs, *outputs)]
                    if len(outputs) != 3 or len(set(pointers)) != 6:
                        raise AssertionError("outputs must own independent storages")
                    for actual, wanted in zip((*outputs, *inputs), (*expected, *arrays)):
                        observed = actual.view(torch.int16).cpu().numpy().view(np.uint16)
                        if not np.array_equal(observed, wanted):
                            raise AssertionError("raw uint16 mismatch")
                    item.update(status="passed", shape=[batch, heads, length, dim],
                                cacheable_key_rows=cache_rows, boundary_mod32=cache_rows * row_bytes % 32,
                                bit_exact=True, inputs_unchanged=True, independent_outputs=True)
                    del outputs, inputs, raw, view, actual, observed
                except Exception as error:
                    item.update(status="failed", error=str(error))
                    report["status"] = "failed"
                    raise
                finally:
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "passed"
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
