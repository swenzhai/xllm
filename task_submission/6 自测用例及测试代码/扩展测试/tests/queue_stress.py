# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Diagnostic stress supplement; does not replace the frozen acceptance suite."""
import argparse
import faulthandler
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--solution-file', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('{"status":"running"}\n')
    faulthandler.dump_traceback_later(60, repeat=True)
    import torch
    import torch_npu
    torch.npu.set_device(0)
    sys.path.insert(0, str(args.package_dir))
    verify = _load('public_verify', args.package_dir/'verify.py')
    golden = _load('golden', args.package_dir/'golden.py')
    gen = _load('generate_data', args.package_dir/'generate_data.py')
    solution = _load('submitted_solution', args.solution_file)
    case = json.loads((args.package_dir/'case.json').read_text())[5]
    attrs = {a['name']: a['value'] for a in case['attr_desc']}
    arrays = gen.generate(case)
    expected = golden.calculate(arrays, attrs)
    report = {'status': 'running', 'records': []}
    for entry in ('run', 'run_direct'):
        fn = getattr(solution, entry)
        inputs = verify.device_inputs(case, arrays, 'npu:0')
        for _ in range(3):
            out = fn(inputs, attrs)
            del out
        torch.npu.synchronize()
        probe = torch.empty(1, device='npu:0')
        del probe
        before = torch.npu.memory_allocated()
        for _ in range(8192):
            out = fn(inputs, attrs)
            del out
        torch.npu.synchronize()
        probe = torch.empty(1, device='npu:0')
        del probe
        after = torch.npu.memory_allocated()
        assert before == after, (before, after)
        del inputs
        compute, pressure_stream = torch.npu.Stream(), torch.npu.current_stream()
        for _ in range(100):
            inputs = verify.device_inputs(case, arrays, 'npu:0')
            owners = [t.untyped_storage() for t in inputs]
            specs = [(tuple(t.shape), t.dtype) for t in inputs]
            torch.npu.synchronize()
            with torch.npu.stream(compute):
                out = fn(inputs, attrs)
            del inputs, owners
            with torch.npu.stream(pressure_stream):
                pressure = [torch.full(shape, 7, dtype=dtype, device='npu:0')
                            for _ in range(8) for shape, dtype in specs]
            compute.synchronize()
            verify.compare(case, arrays, out, expected)
            del out, pressure
        gc.collect()
        torch.npu.synchronize()
        report['records'].append({'entry':entry, 'status':'passed','queued_calls':8192,
                                  'storage_wrapper_release_rounds':100,
                                  'allocated_before':before,'allocated_after':after})
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    report['status'] = 'passed'
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
