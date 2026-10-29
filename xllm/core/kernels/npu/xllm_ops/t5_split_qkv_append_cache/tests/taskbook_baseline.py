# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Taskbook view/concatenation reference, with independently owned query output."""

from typing import Any, Sequence

import torch


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    qkv, past_key, past_value = inputs
    batch, heads, _, dim = past_key.shape
    parts = qkv.reshape(batch, 3, heads, dim)
    query = parts[:, 0].reshape(batch, heads, 1, dim).clone(memory_format=torch.contiguous_format)
    key = parts[:, 1].reshape(batch, heads, 1, dim)
    value = parts[:, 2].reshape(batch, heads, 1, dim)
    return query, torch.cat((past_key, key), dim=2), torch.cat((past_value, value), dim=2)
