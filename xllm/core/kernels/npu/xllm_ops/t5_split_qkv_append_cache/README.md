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

- [x] Shared shape validation and tiling, bit-preserving kernel.
- [x] Private ACLNN operator package and two-stage C++ example.
- [x] PyTorch `run`, `run_direct`, and metadata-only `prepare` interfaces.
- [x] Public and extended correctness, stream/lifetime, memory validation on the development device.
- [x] End-to-end performance measurements and reproducible build evidence on the development device.

The current development checkpoint passes 35 public precision cases and 206
extended records, including both entries, offsets, invalid metadata,
stream/lifetime pressure and allocation recovery. The unchanged official
50/200/5 benchmark passes eight of ten absolute limits. Cases 007 and 017 remain
above their 75/77 microsecond limits. Queue pressure, cache-policy boundaries and
aligned-tail fusion boundaries also pass. See
[the optimization checkpoint](docs/optimization_status.md) for exact measurements,
original evidence and unresolved limits, and [validation details](docs/validation.md)
for reproduction. Earlier passing batches did not reproduce consistently and
are not the current acceptance result. Formal acceptance remains incomplete.

The supplied acceptance package remains external. Its task book and fixed-time
performance files disagree about relative versus absolute timing. Development
uses the supplied fixed limits (28/29/75/77 microseconds), preserving measurements
for clarification. The task-book checksum also differs from its manifest.
Neither discrepancy is resolved by modifying the supplied files.

The available development machine has the correct 9382 SoC and CANN 9.1.0,
but uses Ubuntu 22.04 and driver 26.1.1 rather than the acceptance specification's
openEuler and driver 25.5.t6.b001. Device results must identify this difference.

## Build and run

Use an aarch64 Python 3.12 virtual environment with torch 2.10.0,
torch_npu 2.10.0.post4, NumPy 2.2.6, ml_dtypes 0.6.0, setuptools, wheel, ninja,
and the Python dependencies of CANN. Run these commands from this directory:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh
export T5QKV_INSTALL_PREFIX="$HOME/t5qkv-opp"
bash scripts/install.sh build/custom_opp_ubuntu_aarch64.run "$T5QKV_INSTALL_PREFIX"
source "$T5QKV_INSTALL_PREFIX/vendors/t5qkv/bin/set_env.bash"
export T5QKV_OPAPI_LIB="$T5QKV_INSTALL_PREFIX/vendors/t5qkv/op_api/lib/libcust_opapi.so"
bash scripts/build_framework.sh
```

Use the `.run` filename actually produced for the host OS. The installer refuses
system CANN paths. The direct build uses a fresh build directory on each run
because CANN's legacy kernel preprocessing transforms object files in place.
The extension embeds the direct-library build location in its runtime search
path; rebuild it when moving the project. A complete xLLM server build remains
a separate integration check with the repository's normal dependencies.

The independent C++ example needs no PyTorch:

```bash
cmake -S examples -B build_example -DT5QKV_OPP_PATH="$T5QKV_INSTALL_PREFIX/vendors/t5qkv"
cmake --build build_example
./build_example/t5qkv_aclnn_example
```

The sample invokes both ACLNN stages and checks raw uint16 bits.

Keep the original task package outside this repository. Set `TASK_PACKAGE` to
its absolute directory, then run the official scripts from that directory:

```bash
export T5QKV_SOLUTION="$(pwd)/solution.py"
export T5QKV_REPORTS="$(pwd)/reports"
mkdir -p "$T5QKV_REPORTS"
(cd "$TASK_PACKAGE" && python3 verify.py --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/public_accuracy.json")
python3 tests/acceptance.py --package-dir "$TASK_PACKAGE" --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/extended.json"
python3 tests/queue_stress.py --package-dir "$TASK_PACKAGE" --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/queue_stress.json"
python3 tests/cache_boundary.py --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/cache_boundary.json"
(cd "$TASK_PACKAGE" && python3 benchmark.py --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/performance.json")
python3 tests/compare_same_device.py --package-dir "$TASK_PACKAGE" --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/same_device.json"
(cd "$TASK_PACKAGE" && python3 check_performance.py --report "$T5QKV_REPORTS/performance.json" --output "$T5QKV_REPORTS/performance_acceptance.json")
```

Save stdout/stderr and exit codes under a new run directory on every attempt.
Do not run profiling, lifetime pressure or other NPU workloads concurrently with
performance measurements. `tests/acceptance.py --list --package-dir ...` and
`python3 tests/check_task_mapping.py` run on a CPU host and do not prove NPU
correctness. The `prepare` callable owns only metadata and accepts a fresh input
list with the same shape, dtype and device.


Allocator experiments use the same process configuration for the submission and
reference, exported before torch import. Expose one logical device as required
by the task:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
# Default allocator trial:
unset PYTORCH_NPU_ALLOC_CONF
# Separate expandable allocator trial (use a new report directory):
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python3 tests/tail_fusion_boundary.py --solution-file "$T5QKV_SOLUTION" --report "$T5QKV_REPORTS/tail_fusion_boundary.json"
```

The operator does not modify process configuration. Record the measured setting
with results, and report allocated/active peaks rather than reserved virtual
segments. Earlier trials, including performance failures, remain separate
evidence. The delivery ledger records the actual current status and scope.
