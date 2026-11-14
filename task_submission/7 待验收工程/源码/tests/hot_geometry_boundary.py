# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Bit-exact checks around the constant history-geometry dispatch boundaries."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Choose a new report; original evidence is preserved")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "running", "npu_executed": False, "records": []}
    args.report.write_text(json.dumps(report, indent=2))
    import numpy as np
    import torch
    import torch_npu  # noqa: F401: registers the NPU backend.

    spec = importlib.util.spec_from_file_location("submitted_solution", args.solution_file)
    assert spec is not None and spec.loader is not None
    solution = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = solution
    spec.loader.exec_module(solution)
    torch.npu.set_device(0)
    try:
        for batch, heads, length, dim in [(1, 16, 1472, 64), (1, 16, 1473, 64), (1, 16, 2208, 64),
                                          (1, 16, 2209, 64), (1, 16, 1500, 65), (1, 16, 2048, 48),
                                          (1, 15, 1800, 64)]:
            shapes = [(batch, 1, 3 * heads * dim), (batch, heads, length, dim)]
            shapes.append(shapes[1])
            arrays = [(np.arange(np.prod(shape), dtype=np.uint16) + index * 997).reshape(shape)
                      for index, shape in enumerate(shapes)]
            packed = arrays[0].reshape(batch, 3, heads, dim)
            expected = [packed[:, 0, :, None, :].copy()]
            expected += [np.concatenate((arrays[i], packed[:, i, :, None, :]), axis=2) for i in (1, 2)]
            for entry in ("run", "run_direct"):
                for dtype in (torch.float16, torch.bfloat16):
                    for offset in (0, 1):
                        inputs = []
                        for array in arrays:
                            owner = torch.empty(array.size + offset, dtype=torch.int16, device="npu:0")
                            view = owner[offset:].reshape(array.shape)
                            view.copy_(torch.from_numpy(array.view(np.int16)))
                            inputs.append(view.view(dtype))
                        outputs = tuple(getattr(solution, entry)(inputs, {"headNum": heads, "headDim": dim}))
                        torch.npu.synchronize()
                        report["npu_executed"] = True
                        assert len(outputs) == 3 and all(t.is_contiguous() for t in outputs)
                        pointers = [t.untyped_storage().data_ptr() for t in (*inputs, *outputs)]
                        assert len(set(pointers)) == 6
                        for actual, wanted in zip((*outputs, *inputs), (*expected, *arrays)):
                            observed = actual.view(torch.int16).cpu().numpy().view(np.uint16)
                            assert np.array_equal(observed, wanted), (entry, dtype, offset)
                        record = {"shape": [batch, heads, length, dim], "entry": entry,
                                  "dtype": str(dtype), "storage_offset": offset, "status": "passed",
                                  "bit_exact": True, "inputs_unchanged": True, "independent_outputs": True}
                        report["records"].append(record)
                        print(json.dumps(record), flush=True)
                        args.report.write_text(json.dumps(report, indent=2))
                        del outputs, inputs, owner, view, actual, observed
        assert len(report["records"]) == 56
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
