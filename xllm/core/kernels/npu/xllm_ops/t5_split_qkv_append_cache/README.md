<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# T5 QKV split and cache append

This standalone Ascend C project implements `T5SplitQkvAppendCache` for
Ascend910_9382 (Atlas A3), CANN 9.1.0. It splits a single-token packed QKV tensor,
copies the previous K/V caches, and appends the current token. All three outputs
own separate contiguous storage. Both float16 and bfloat16 preserve every input
bit, including NaN payloads. No global workspace is allowed.

The development branch starts at xLLM commit
`3967b8a20adb238fd6302020563814d5e5c03493`.
The project lives beside the xLLM NPU wrappers so it can be built and validated
without building the full inference server or changing the xllm_ops submodule.

## Development milestones

- [ ] Shared shape validation and tiling, bit-preserving kernel.
- [ ] Private ACLNN operator package and two-stage C++ example.
- [ ] PyTorch `run`, `run_direct`, and metadata-only `prepare` interfaces.
- [ ] Public and extended correctness, stream/lifetime, memory validation.
- [ ] End-to-end performance measurements and reproducible build evidence.

The supplied acceptance package remains external. Its task book and fixed-time
performance files disagree about relative versus absolute timing. Development
uses the supplied fixed limits (28/29/75/77 microseconds), preserving measurements
for clarification. The task-book checksum also differs from its manifest.
Neither discrepancy is resolved by modifying the supplied files.

The available development machine has the correct 9382 SoC and CANN 9.1.0,
but uses Ubuntu 22.04 and driver 26.1.1 rather than the acceptance specification's
openEuler and driver 25.5.t6.b001. Device results must identify this difference.
