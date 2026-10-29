#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extended T5SplitQkvAppendCache device checks against an external task package.

The source package is read only. This script does not replace its public verifier
or fixed-protocol benchmark. ``--list`` validates the coverage inventory without
importing torch, NumPy, the solution, or any code from the task package.
"""
import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
import traceback
import weakref
from typing import Any, Callable
from types import ModuleType
from pathlib import Path

GROUPS = ("public", "extra", "negative", "metadata", "verifier", "streams",
          "input_lifetime", "output_lifetime", "recovery", "append", "memory")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inventory(package: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cases = json.loads((package / "case.json").read_text())
    coverage = json.loads((package / "coverage_requirements.json").read_text())
    targets = json.loads((package / "performance_targets.json").read_text())
    by_id = {case["case_name"]: case for case in cases}
    if len(cases) != 35 or len(by_id) != len(cases):
        raise ValueError("expected 35 distinct public cases")
    if set(by_id) != set(coverage["public_cases"]):
        raise ValueError("public case IDs disagree with coverage_requirements.json")
    for name, expected in coverage["public_cases"].items():
        payload = json.dumps(by_id[name], sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("public case definition changed: " + name)
    if _sha256(package / "case.json") != targets["case_sha256"]:
        raise ValueError("public case file hash disagrees with performance targets")
    performance = {c["case_name"] for c in cases if c.get("performance") is True}
    if len(performance) != 10 or performance != set(coverage["performance_case_ids"]):
        raise ValueError("expected the ten frozen performance cases")
    if len(coverage["negative_scenarios"]) != 28:
        raise ValueError("expected 28 negative scenarios")
    if coverage["negative_entries"] != ["prepare", "run", "run_direct"]:
        raise ValueError("unexpected negative entry coverage")
    if len(coverage["extra_input_case_ids"]) != 12:
        raise ValueError("expected 12 extra inputs")
    if coverage["extra_entries"] != ["run", "run_direct"]:
        raise ValueError("unexpected extra entry coverage")
    for name in coverage["extra_input_case_ids"]:
        if name.split("__extra_seed_", 1)[0] not in by_id:
            raise ValueError("extra input lacks public source: " + name)
    return cases, by_id, coverage


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _attributes(case: dict[str, Any]) -> dict[str, Any]:
    return {a["name"]: a["value"] for a in case["attr_desc"]}


def _expected_rejection(fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = fn()
        del value
    except (TypeError, ValueError, RuntimeError, AssertionError) as error:
        return {"rejected": True, "exception": type(error).__name__, "message": str(error)}
    raise AssertionError("invalid input was accepted")


class Suite:
    def __init__(self, args: Any, cases: Any, by_id: Any, coverage: Any) -> None:
        import numpy as np
        import torch
        import torch_npu
        self._np, self._torch, self._torch_npu = np, torch, torch_npu
        self._args, self._cases, self._by_id, self._coverage = args, cases, by_id, coverage
        self._device = f"npu:{args.device}"
        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; no device tests executed")
        torch.npu.set_device(args.device)
        # Avoid writing __pycache__ into the frozen package or solution directory.
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.package_dir))
        self._golden = _load_module("golden", args.package_dir / "golden.py")
        self._generator = _load_module("generate_data", args.package_dir / "generate_data.py")
        self._public_verifier = _load_module("t5_public_verifier", args.package_dir / "verify.py")
        sys.path.insert(0, str(args.solution_file.parent))
        self._solution = _load_module("t5_submitted_solution", args.solution_file)
        for name in ("run", "run_direct", "prepare"):
            if not callable(getattr(self._solution, name, None)):
                raise TypeError("solution must expose callable " + name)
        self._report = {
            "schema_version": 2, "status": "running", "npu_executed": False,
            "scope": "extended_checks_only", "selected_groups": args.groups,
            "package_dir": str(args.package_dir), "solution_file": str(args.solution_file),
            "case_sha256": _sha256(args.package_dir / "case.json"),
            "solution_sha256": _sha256(args.solution_file),
            "test_script_sha256": _sha256(Path(__file__)),
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "numpy": np.__version__, "device": self._device,
                "device_name": torch.npu.get_device_name(args.device),
                "formal_environment_verified": False,
            }, "records": [],
        }
        self._flush()

    def _flush(self) -> None:
        self._args.report.parent.mkdir(parents=True, exist_ok=True)
        temp = self._args.report.with_suffix(self._args.report.suffix + ".tmp")
        temp.write_text(json.dumps(self._report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temp.replace(self._args.report)

    def _record(self, group: str, name: str, entry: str, fn: Callable[[], Any]) -> None:
        started = time.monotonic()
        item = {"group": group, "case_name": name, "entry": entry, "status": "running"}
        self._report["records"].append(item)
        self._flush()
        try:
            details = fn()
            item.update(status="passed", details=details)
        except Exception as error:
            item.update(status="failed", error=str(error), exception=type(error).__name__,
                        traceback=traceback.format_exc())
        except BaseException as error:
            item.update(status="interrupted", error=str(error), exception=type(error).__name__)
            raise
        finally:
            item["elapsed_seconds"] = time.monotonic() - started
            print(json.dumps(item, ensure_ascii=False, allow_nan=False), flush=True)
            self._flush()
        if item["status"] == "failed" and self._args.fail_fast:
            raise RuntimeError(f"{group}/{name}/{entry} failed: {item['error']}")

    def _inputs(self, case: Any, offset: Any = False) -> Any:
        arrays = self._generator.generate(case)
        inputs = self._public_verifier.device_inputs(case, arrays, self._device)
        if offset:
            result = []
            for value in inputs:
                storage = self._torch.empty(value.numel() + 16, dtype=value.dtype, device=self._device)
                view = storage[1:1 + value.numel()].view(value.shape)
                # Move the raw encoding, including signalling NaNs, without casting.
                view.view(self._torch.int16).copy_(value.view(self._torch.int16))
                result.append(view)
            inputs = result
        return arrays, inputs

    def _check(self, case: Any, arrays: Any, inputs: Any, outputs: Any, expected: Any = None) -> Any:
        outputs = tuple(outputs)
        self._torch.npu.synchronize()
        self._report["npu_executed"] = True
        self._public_verifier.validate_ownership(inputs, outputs)
        expected = self._golden.calculate(arrays, _attributes(case)) if expected is None else expected
        checks = self._public_verifier.compare(case, arrays, outputs, expected)
        for tensor, array in zip(inputs, arrays):
            if not self._np.array_equal(self._public_verifier.cpu_array(tensor).view(self._np.uint16),
                                       array.view(self._np.uint16)):
                raise AssertionError("input bit pattern changed")
        return {"outputs": checks, "input_unchanged": True,
                "output_storage": [t.untyped_storage().data_ptr() for t in outputs]}

    def _accuracy(self, case: Any, entry: Any, offset: Any = False) -> Any:
        arrays, inputs = self._inputs(case, offset)
        outputs = getattr(self._solution, entry)(inputs, _attributes(case))
        result = self._check(case, arrays, inputs, outputs)
        result["input_storage_offsets"] = [t.storage_offset() for t in inputs]
        return result

    def _public(self) -> None:
        for case in self._cases:
            for entry in ("run", "run_direct"):
                self._record("public", case["case_name"], entry,
                            lambda c=case, e=entry: self._accuracy(c, e))

    def _extra(self) -> None:
        for name in self._coverage["extra_input_case_ids"]:
            case = copy.deepcopy(self._by_id[name.split("__extra_seed_", 1)[0]])
            case.update(case_name=name, kind="random", performance=False)
            for entry in self._coverage["extra_entries"]:
                self._record("extra", name, entry, lambda c=case, e=entry: self._accuracy(c, e, True))

    def _negative_inputs(self, scenario: str) -> tuple[list[Any], Any]:
        case = self._by_id[self._coverage["performance_case_ids"][0]]
        _, inputs = self._inputs(case)
        attrs = _attributes(case)
        torch = self._torch
        q, pk, pv = inputs
        if scenario.startswith(("headNum_", "headDim_")):
            key, literal = scenario.split("_", 1)
            attrs[key] = ast.literal_eval(literal)
        elif scenario.startswith("shape_limit_"):
            b, h, l, d = ast.literal_eval(scenario[len("shape_limit_"):])
            inputs = [torch.empty((b, 1, 3*h*d), dtype=q.dtype, device=self._device),
                      torch.empty((b, h, l, d), dtype=q.dtype, device=self._device),
                      torch.empty((b, h, l, d), dtype=q.dtype, device=self._device)]
            attrs = {}
        elif scenario == "attributes_type": attrs = []
        elif scenario == "cache_dtype": inputs[1] = pk.float()
        elif scenario == "cache_mismatch":
            b, h, l, d = pv.shape
            inputs[2] = torch.empty((b, h, l+1, d), dtype=pv.dtype, device=self._device)
        elif scenario == "cache_rank": inputs[1] = torch.empty((2, 3), dtype=q.dtype, device=self._device)
        elif scenario == "cpu_input": inputs[0] = q.cpu()
        elif scenario == "float32": inputs[0] = q.float()
        elif scenario == "input_count": inputs = inputs[:-1]
        elif scenario == "noncontiguous":
            storage = torch.empty((*q.shape[:-1], 2*q.shape[-1]), dtype=q.dtype, device=self._device)
            inputs[0] = storage[..., ::2]
            assert not inputs[0].is_contiguous()
        elif scenario == "packed_mismatch":
            inputs[0] = torch.empty((*q.shape[:-1], q.shape[-1]+1), dtype=q.dtype, device=self._device)
        elif scenario == "packed_tokens":
            inputs[0] = torch.empty((q.shape[0], 2, q.shape[-1]), dtype=q.dtype, device=self._device)
        elif scenario == "unknown_attribute": attrs["unknown"] = 1
        else: raise ValueError("unimplemented negative scenario: " + scenario)
        return inputs, attrs

    def _negative(self) -> None:
        for scenario in self._coverage["negative_scenarios"]:
            for entry in self._coverage["negative_entries"]:
                def _run(scenario: Any = scenario, entry: Any = entry) -> Any:
                    inputs, attrs = self._negative_inputs(scenario)
                    return _expected_rejection(lambda: getattr(self._solution, entry)(inputs, attrs))
                self._record("negative", scenario, entry, _run)

    def _metadata(self) -> None:
        # This hook must invoke the same bottom-level output validator as execution.
        def _trial(kind: Any) -> Any:
            validate = getattr(self._solution, "validate_out", None)
            if not callable(validate):
                raise RuntimeError("solution.validate_out(inputs, attributes, outputs) is required")
            case = self._by_id[self._coverage["performance_case_ids"][1]]
            arrays, inputs = self._inputs(case)
            attrs = _attributes(case)
            outputs = [self._torch.empty(desc["shape"], dtype=inputs[0].dtype, device=self._device)
                       for desc in case["output_desc"]]
            if kind == "output_count": outputs = outputs[:-1]
            elif kind == "output_shape": outputs[0] = outputs[0].flatten()
            elif kind == "output_dtype": outputs[1] = outputs[1].float()
            elif kind == "output_device": outputs[2] = outputs[2].cpu()
            elif kind == "output_storage_alias": outputs[2] = outputs[1]
            elif kind == "output_noncontiguous":
                desc = case["output_desc"][0]["shape"]
                storage = self._torch.empty((*desc[:-1], desc[-1]*2), dtype=inputs[0].dtype,
                                           device=self._device)
                outputs[0] = storage[..., ::2]
            return _expected_rejection(lambda: validate(inputs, attrs, outputs))
        for kind in ("output_count", "output_shape", "output_dtype", "output_device",
                     "output_storage_alias", "output_noncontiguous"):
            self._record("metadata", kind, "validate_out", lambda k=kind: _trial(k))

    def _verifier(self) -> None:
        def _trial(kind: Any) -> Any:
            case = self._by_id[self._coverage["performance_case_ids"][0]]  # L=0: equal output sizes.
            arrays, inputs = self._inputs(case)
            expected = self._golden.calculate(arrays, _attributes(case))
            outputs = list(self._solution.run(inputs, _attributes(case)))
            self._torch.npu.synchronize()
            if kind == "CPU output rejected": outputs[0] = outputs[0].cpu()
            elif kind == "input storage reuse rejected":
                b, h, _, d = case["output_desc"][0]["shape"]
                outputs[0] = inputs[0].view(b, 3, h, d)[:, 0].reshape(b, h, 1, d)
            elif kind == "offset overlapping storage rejected":
                count = outputs[1].numel()
                shared = self._torch.empty(2*count+1, dtype=outputs[1].dtype, device=self._device)
                outputs[1] = shared[:count].view(outputs[1].shape)
                outputs[2] = shared[1:count+1].view(outputs[2].shape)
                assert outputs[1].data_ptr() != outputs[2].data_ptr()
            def _validate() -> Any:
                self._public_verifier.validate_ownership(inputs, outputs)
                return self._public_verifier.compare(case, arrays, outputs, expected)
            return _expected_rejection(_validate)
        for kind in self._coverage["verifier_checks"]:
            self._record("verifier", kind, "public_verifier", lambda k=kind: _trial(k))

    def _streams(self) -> None:
        def _run() -> Any:
            torch = self._torch
            streams = [torch.npu.Stream(device=self._args.device) for _ in range(2)]
            cases = [self._by_id[name] for name in self._coverage["stream_case_ids"]]
            data = [self._inputs(case) for case in cases]
            plans = [self._solution.prepare(inputs, _attributes(case))
                     for case, (_, inputs) in zip(cases, data)]
            if not all(callable(plan) for plan in plans):
                raise TypeError("prepare must return callable(inputs)")
            torch.npu.synchronize()
            outputs = [[], []]
            for _ in range(self._coverage["min_prepared_calls_per_stream"]):
                for i in range(2):
                    with torch.npu.stream(streams[i]):
                        outputs[i].append(plans[i](data[i][1]))
            results = []
            for i in range(2):
                streams[i].synchronize()
                arrays, inputs = data[i]
                for actual in outputs[i]:
                    results.append({"case_name": cases[i]["case_name"], "stream_index": i,
                                    "checks": self._check(cases[i], arrays, inputs, actual)})
            all_ptrs = [t.untyped_storage().data_ptr() for group in outputs for out in group for t in out]
            if len(set(all_ptrs)) != len(all_ptrs):
                raise AssertionError("prepared calls reuse output storage")
            return {"streams": 2, "prepared_calls_per_stream": len(outputs[0]), "checks": results}
        self._record("streams", "independent_streams", "prepare/run", _run)

    def _input_lifetime(self) -> None:
        def _run(entry: Any) -> Any:
            torch = self._torch
            case = self._by_id[self._coverage["stream_case_ids"][1]]
            compute = torch.npu.Stream(device=self._args.device)
            pressure_stream = torch.npu.current_stream()
            # Queue a known-valid workload before the short copy kernel. Without
            # a backlog, the kernel can finish before Python releases inputs,
            # making an otherwise successful pressure test inconclusive.
            delay_input = torch.ones((4096, 4096), dtype=torch.float16, device=self._device)
            delay_output = torch.empty_like(delay_input)
            torch.npu.synchronize()
            with torch.npu.stream(compute):
                torch.mm(delay_input, delay_input, out=delay_output)
            compute.synchronize()  # Compilation/warmup is outside the stress interval.
            observed = []
            for _ in range(8):
                arrays, inputs = self._inputs(case)  # Allocation belongs to default stream.
                expected = self._golden.calculate(arrays, _attributes(case))
                torch.npu.synchronize()  # Finish input transfer, before async invocation.
                refs = [weakref.ref(t) for t in inputs]
                specs = [(tuple(t.shape), t.dtype) for t in inputs]
                input_storage = {t.untyped_storage().data_ptr() for t in inputs if t.numel()}
                with torch.npu.stream(compute):
                    for _ in range(8):
                        torch.mm(delay_input, delay_input, out=delay_output)
                    outputs = getattr(self._solution, entry)(inputs, _attributes(case))
                pending_at_release = not compute.query()
                del inputs
                # Do not run the cyclic collector here: its host delay can let
                # the kernel finish before the cross-stream pressure starts.
                # Framework host queues may legitimately retain tensors until
                # submission. The caller owns no strong reference from here on.
                live_at_release = sum(ref() is not None for ref in refs)
                with torch.npu.stream(pressure_stream):
                    pressure = [torch.empty(shape, dtype=dtype, device=self._device)
                                for _ in range(16) for shape, dtype in specs]
                    for tensor in pressure:
                        tensor.fill_(7)
                    del tensor
                compute.synchronize()
                output_storage = [t.untyped_storage().data_ptr() for t in outputs]
                if len(set(output_storage)) != 3 or input_storage.intersection(output_storage):
                    raise AssertionError("released input or output storage is aliased")
                checks = self._public_verifier.compare(case, arrays, outputs, expected)
                pressure_stream.synchronize()
                del outputs, pressure
                self._settle_allocator()
                live_after_sync = sum(ref() is not None for ref in refs)
                if live_after_sync:
                    raise AssertionError(f"{live_after_sync} input Tensor objects remain after completion")
                observed.append({"pending_at_release": pending_at_release,
                                 "live_references_at_release": live_at_release,
                                 "live_references_after_sync": live_after_sync,
                                 "checks": checks})
            self._report["npu_executed"] = True
            pending_count = sum(item["pending_at_release"] for item in observed)
            if pending_count == 0:
                raise AssertionError("input-release stress was inconclusive: no pending device work at release")
            return {"repetitions": 8, "allocations_per_repetition": 48,
                    "backlog": "8 FP16 4096x4096 matmuls before each submitted operator",
                    "pending_at_release_count": pending_count,
                    "compute_stream": str(compute), "pressure_stream": str(pressure_stream),
                    "outputs": observed}
        for entry in ("run", "run_direct"):
            self._record("input_lifetime", self._coverage["stream_case_ids"][1], entry,
                        lambda e=entry: _run(e))

    def _output_lifetime(self) -> None:
        def _run(entry: Any) -> Any:
            case = self._by_id[self._coverage["stream_case_ids"][0]]
            arrays, inputs = self._inputs(case)
            attrs = _attributes(case)
            target_stream = self._torch.npu.current_stream()
            retained = [getattr(self._solution, entry)(inputs, attrs) for _ in range(8)]
            ptrs = [t.untyped_storage().data_ptr() for out in retained for t in out]
            if len(ptrs) != len(set(ptrs)):
                raise AssertionError("retained output storages are not independent")
            specs = [(tuple(t.shape), t.dtype) for t in retained[0]]
            released_refs = []
            live_at_release = []
            for _ in range(32):
                if self._torch.npu.current_stream() != target_stream:
                    raise AssertionError("output pressure stream changed")
                temporary = getattr(self._solution, entry)(inputs, attrs)
                refs = [weakref.ref(t) for t in temporary]
                del temporary
                # Retaining a short-lived host-queue reference is legal; verify
                # actual cleanup only after all queued work has completed.
                live_at_release.append(sum(ref() is not None for ref in refs))
                released_refs.extend(refs)
                pressure = [self._torch.empty(shape, dtype=dtype, device=self._device)
                            for shape, dtype in specs]
                for tensor in pressure:
                    tensor.fill_(11)
                del tensor, pressure
            checks = [self._check(case, arrays, inputs, outputs) for outputs in retained]
            target_stream.synchronize()
            self._settle_allocator()
            live_after_sync = sum(ref() is not None for ref in released_refs)
            if live_after_sync:
                raise AssertionError(f"{live_after_sync} released output Tensor objects remain after completion")
            return {"retained_calls": 8, "release_pressure_calls": 32,
                    "live_references_at_release": live_at_release,
                    "live_references_after_sync": live_after_sync,
                    "target_and_pressure_stream": str(target_stream), "checks": checks}
        for entry in ("run", "run_direct"):
            self._record("output_lifetime", self._coverage["stream_case_ids"][0], entry,
                        lambda e=entry: _run(e))

    def _allocated(self) -> int:
        return int(self._torch.npu.memory_allocated(self._args.device))

    def _active(self, peak: bool = False) -> int:
        stats = self._torch.npu.memory_stats(self._args.device)
        key = "active_bytes.all.peak" if peak else "active_bytes.all.current"
        if key not in stats:
            raise RuntimeError("allocator does not expose " + key)
        return int(stats[key])

    def _settle_allocator(self) -> None:
        gc.collect()
        self._torch.npu.synchronize()
        # recordStream frees may await event processing even after device sync.
        # A tiny allocation polls completed events; it is outside the measured region.
        probe = self._torch.empty(1, dtype=self._torch.uint8, device=self._device)
        del probe
        self._torch.npu.synchronize()

    def _recovery(self) -> None:
        def _run(entry: Any) -> Any:
            case = self._by_id[self._coverage["stream_case_ids"][1]]
            _, inputs = self._inputs(case)
            fn = getattr(self._solution, entry)
            attrs = _attributes(case)
            for _ in range(3):
                out = fn(inputs, attrs)
                self._torch.npu.synchronize()
                del out
            self._settle_allocator()
            before, active_before = self._allocated(), self._active()
            for _ in range(20):
                out = fn(inputs, attrs)
                del out
            self._settle_allocator()
            after, active_after = self._allocated(), self._active()
            if after != before or active_after != active_before:
                raise AssertionError(f"active tensor leak: allocated {before}->{after}; "
                                     f"active {active_before}->{active_after}")
            self._report["npu_executed"] = True
            return {"calls": 20, "allocated_before": before, "allocated_after": after,
                    "active_before": active_before, "active_after": active_after}
        for entry in ("run", "run_direct"):
            self._record("recovery", self._coverage["stream_case_ids"][1], entry,
                        lambda e=entry: _run(e))

    def _append(self) -> None:
        def _run(entry: Any) -> Any:
            case = next(c for c in self._cases if c["input_desc"][1]["shape"][2] == 31)
            arrays, inputs = self._inputs(case)
            attrs = _attributes(case)
            first = getattr(self._solution, entry)(inputs, attrs)
            self._check(case, arrays, inputs, first)
            second_inputs = [inputs[0], first[1], first[2]]
            second_arrays = [arrays[0], self._public_verifier.cpu_array(first[1]).copy(),
                             self._public_verifier.cpu_array(first[2]).copy()]
            second_case = copy.deepcopy(case)
            second_case["case_name"] += "__second_append"
            for desc in second_case["input_desc"][1:]: desc["shape"][2] += 1
            for desc in second_case["output_desc"][1:]: desc["shape"][2] += 1
            second = getattr(self._solution, entry)(second_inputs, attrs)
            return self._check(second_case, second_arrays, second_inputs, second)
        for entry in ("run", "run_direct"):
            self._record("append", "two_consecutive_appends", entry, lambda e=entry: _run(e))

    def _memory(self) -> None:
        def _run(case: Any) -> Any:
            _, inputs = self._inputs(case)
            attrs = _attributes(case)
            workspace_size = getattr(self._solution, "workspace_size", None)
            if not callable(workspace_size):
                raise RuntimeError("solution.workspace_size(inputs, attributes) required for workspace evidence")
            workspace = int(workspace_size(inputs, attrs))
            if workspace != 0:
                raise AssertionError(f"explicit global workspace must be zero, got {workspace}")
            for _ in range(3):
                outputs = self._solution.run(inputs, attrs)
                self._torch.npu.synchronize()
                del outputs
            self._settle_allocator()
            before = self._allocated()
            active_before = self._active()
            self._torch.npu.reset_peak_memory_stats(self._args.device)
            outputs = self._solution.run(inputs, attrs)
            self._torch.npu.synchronize()
            peak = int(self._torch.npu.max_memory_allocated(self._args.device))
            active_peak = self._active(peak=True)
            output_bytes = sum(t.numel()*t.element_size() for t in outputs)
            delta, limit = peak-before, output_bytes+16384
            active_delta = active_peak-active_before
            if delta > limit or active_delta > limit:
                raise AssertionError(f"peak deltas allocated={delta}, active={active_delta} exceed {limit}")
            del outputs
            self._settle_allocator()
            after, active_after = self._allocated(), self._active()
            if after != before or active_after != active_before:
                raise AssertionError("active tensor allocation did not return to baseline")
            self._report["npu_executed"] = True
            return {"explicit_workspace_bytes": workspace, "output_bytes": output_bytes,
                    "peak_allocated_delta": delta, "peak_active_delta": active_delta,
                    "allocation_metric": "memory_allocated", "active_metric": "active_bytes.all",
                    "limit_bytes": limit,
                    "allocated_before": before, "allocated_after": after,
                    "active_before": active_before, "active_after": active_after}
        for name in self._coverage["performance_case_ids"]:
            self._record("memory", name, "run", lambda c=self._by_id[name]: _run(c))

    def execute(self) -> int:
        expected = {"public": 2*len(self._cases), "extra": 24, "negative": 84,
                    "metadata": 6, "verifier": 3, "streams": 1, "input_lifetime": 2,
                    "output_lifetime": 2, "recovery": 2, "append": 2, "memory": 10}
        self._report["expected_records"] = sum(expected[group] for group in self._args.groups)
        self._report["completed_groups"] = []
        completed = False
        try:
            for group in self._args.groups:
                self._report["active_group"] = group
                self._flush()
                getattr(self, "_" + group)()
                self._report["completed_groups"].append(group)
            completed = True
        finally:
            records = self._report["records"]
            failed = [r for r in records if r["status"] != "passed"]
            coverage_complete = completed and len(records) == self._report["expected_records"]
            self._report["status"] = "passed" if coverage_complete and not failed else "failed"
            self._report["coverage_complete"] = coverage_complete
            self._report["failed_records"] = len(failed)
            self._report["passed_records"] = len(records)-len(failed)
            self._report["all_groups_selected"] = set(self._args.groups) == set(GROUPS)
            self._report["performance_measured"] = False
            if completed:
                self._report.pop("active_group", None)
            self._flush()
        return 0 if self._report["status"] == "passed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--solution-file", type=Path)
    parser.add_argument("--report", type=Path, default=Path("extended_checks.json"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--groups", default=",".join(GROUPS), help="comma-separated: " + ",".join(GROUPS))
    parser.add_argument("--list", action="store_true", help="validate/list frozen inventory without importing torch")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    args.package_dir = args.package_dir.resolve()
    if args.list:
        cases, by_id, coverage = _inventory(args.package_dir)
        print(json.dumps({"status": "inventory_valid", "npu_executed": False,
                          "public_inputs": len(cases), "public_records_both_entries": 2*len(cases),
                          "extra_inputs": len(coverage["extra_input_case_ids"]),
                          "extra_records": 24, "negative_scenarios": 28, "negative_records": 84,
                          "performance_inputs": len(coverage["performance_case_ids"]),
                          "groups": GROUPS, "public_case_ids": list(by_id)}, indent=2))
        return 0
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"status": "initializing", "npu_executed": False,
                                      "started_unix": time.time()}) + "\n")
    try:
        if args.solution_file is None:
            raise ValueError("--solution-file is required unless --list is used")
        args.solution_file = args.solution_file.resolve()
        args.groups = args.groups.split(",")
        if len(set(args.groups)) != len(args.groups) or any(g not in GROUPS for g in args.groups):
            raise ValueError("--groups must contain distinct known group names")
        cases, by_id, coverage = _inventory(args.package_dir)
        return Suite(args, cases, by_id, coverage).execute()
    except BaseException as error:
        # Process boundary: preserve partial records and never report an interrupt as a pass.
        result = json.loads(args.report.read_text())
        result.update(status="failed", error=str(error), exception=type(error).__name__,
                      traceback=traceback.format_exc())
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        # stdout is a machine-readable result stream, not diagnostic logging.
        print(json.dumps({"status": "failed", "error": str(error), "report": str(args.report)}))
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
