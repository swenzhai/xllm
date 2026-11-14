# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Acceptance entry points; build the extension before importing this module."""

import importlib.util
from pathlib import Path
from typing import Any, Sequence

import torch
import torch_npu  # noqa: F401: registers the NPU backend.

_EXTENSION: Any = None


def _extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        candidates = list(Path(__file__).parent.glob("_t5qkv*.so"))
        if len(candidates) != 1:
            raise RuntimeError("Build the T5QKV extension first; expected one _t5qkv*.so")
        spec = importlib.util.spec_from_file_location("_t5qkv", candidates[0])
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the T5QKV extension")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXTENSION = module
    return _EXTENSION


def _arguments(inputs: Sequence[torch.Tensor], attributes: dict) -> tuple[int, int]:
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
        raise ValueError("inputs must contain qkv, pastKey, pastValue")
    if not isinstance(attributes, dict):
        raise TypeError("attributes must be a dictionary")
    if attributes.keys() - {"headNum", "headDim"}:
        raise ValueError("unknown T5QKV attribute")
    head_num = attributes.get("headNum", 1)
    head_dim = attributes.get("headDim", 1)
    if type(head_num) is not int or type(head_dim) is not int:
        raise TypeError("headNum and headDim must be integers, excluding bool")
    if not 1 <= head_num <= 128 or not 1 <= head_dim <= 256:
        raise ValueError("headNum or headDim is outside the supported range")
    return head_num, head_dim


def run(inputs: Sequence[torch.Tensor], attributes: dict) -> tuple[torch.Tensor, ...]:
    """Allocate independent outputs and invoke the two-stage ACLNN API."""
    head_num, head_dim = _arguments(inputs, attributes)
    return tuple(_extension().run(inputs, head_num, head_dim, False))


def run_direct(inputs: Sequence[torch.Tensor], attributes: dict) -> tuple[torch.Tensor, ...]:
    """Allocate independent outputs and launch the shared Ascend C kernel."""
    head_num, head_dim = _arguments(inputs, attributes)
    return tuple(_extension().run(inputs, head_num, head_dim, True))


class _Prepared:
    """A reusable metadata contract; owns no tensors, outputs, or stream."""

    def __init__(self, metadata: tuple, head_num: int, head_dim: int) -> None:
        self._metadata = metadata
        self._head_num = head_num
        self._head_dim = head_dim

    def __call__(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        attrs = {"headNum": self._head_num, "headDim": self._head_dim}
        _arguments(inputs, attrs)
        metadata = tuple(_extension().validate(inputs, self._head_num, self._head_dim))
        if metadata != self._metadata:
            raise ValueError("prepared input metadata changed")
        return tuple(_extension().run(inputs, self._head_num, self._head_dim, False))


def prepare(inputs: Sequence[torch.Tensor], attributes: dict) -> _Prepared:
    """Validate once and return callable(new_inputs); calls still allocate outputs."""
    head_num, head_dim = _arguments(inputs, attributes)
    metadata = tuple(_extension().validate(inputs, head_num, head_dim))
    return _Prepared(metadata, head_num, head_dim)


def validate_out(
    inputs: Sequence[torch.Tensor], attributes: dict, outputs: Sequence[torch.Tensor]
) -> None:
    """Expose the extension's output metadata/storage checks without launching."""
    head_num, head_dim = _arguments(inputs, attributes)
    _extension().validate_out(inputs, outputs, head_num, head_dim)


def workspace_size(inputs: Sequence[torch.Tensor], attributes: dict) -> int:
    """Static workspace contract, also enforced against each real ACLNN query."""
    head_num, head_dim = _arguments(inputs, attributes)
    _extension().validate(inputs, head_num, head_dim)
    return 0
