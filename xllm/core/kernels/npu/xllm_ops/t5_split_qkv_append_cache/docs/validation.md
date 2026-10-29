<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Development validation: 2026-10-29


## Current optimized validation (2026-10-30)

The current implementation uses 47,104-element history tiles, two 92 KiB UB
buffers, one-task-ahead prefetch, a 172 MiB L2 admission budget, and guarded UB
tail fusion. No output caching, global workspace or arithmetic conversion is
introduced. The fallback retains separate append tasks for unsupported tails.
`build_framework.sh --force` ensures shared-header changes rebuild the extension.

The fixed run `optimized_tail_buffer172_final_01` passed 35 public accuracy
cases, all 206 extended records (including ten active-allocation measurements),
8,192 queued calls plus 100 lifetime-pressure rounds per entry, eight unaligned
cache-boundary checks and 80 host shape/core/offset checks (165,960 tasks).
All four complete public-timer batches passed the ten absolute limits. Both
reference qualifications passed all 35 public cases. Source and installed binary
manifests were identical before and after the entire run. The subsequent C++
example also passed FP16 and BF16 with bit-exact output and zero workspace.

| Case ID | Minimum us across four batches | Maximum us across four batches | Limit us |
| --- | ---: | ---: | ---: |
| T5SplitQkvAppendCache_001_float16_random | 16.08180 | 17.44225 | 28 |
| T5SplitQkvAppendCache_005_float16_random | 16.77565 | 17.81520 | 28 |
| T5SplitQkvAppendCache_006_float16_random | 16.69660 | 17.67030 | 29 |
| T5SplitQkvAppendCache_007_float16_random | 68.04515 | 74.64870 | 75 |
| T5SplitQkvAppendCache_008_float16_random | 17.30555 | 19.23500 | 29 |
| T5SplitQkvAppendCache_011_bfloat16_random | 16.92785 | 18.77670 | 28 |
| T5SplitQkvAppendCache_015_bfloat16_random | 17.78955 | 19.27860 | 28 |
| T5SplitQkvAppendCache_016_bfloat16_random | 17.75165 | 20.03345 | 29 |
| T5SplitQkvAppendCache_017_bfloat16_random | 67.28030 | 74.92840 | 77 |
| T5SplitQkvAppendCache_018_bfloat16_random | 17.93245 | 20.09875 | 29 |

Every batch and every round is retained; the table gives a range and is not a
best-batch selection. Delivery tables use the complete final comparison batch
`same_device_confirm.json`. Prior candidates and the non-fused 172 MiB version
had repeat failures and are not promoted as the final result. The previous
baseline-comparison prototype also had a module-path error; its original child
measurements and failed driver remain archived.

Measured hardware remains Ascend910_9382, CANN 9.1.0, Ubuntu 22.04.5 and driver
26.1.1. The task's specified OS/driver/single-visible-device environment, ambiguous
ratio-versus-microsecond column, original task-manifest anomaly, full xLLM server
integration, TBE applicability and external community submission remain explicit
acceptance gaps. These device results do not assert formal acceptance.

## Historical implementation and evidence

The following sections describe the previous implementation and are retained for
traceability. Current source, timing and UB parameters are those above and in
`design.md`; historical report values are not substituted for the current run.

Tested implementation: `ab70af7891455b36c7aaac6344f0f2207afcc068`, on branch
`codex/t5-qkv-split-cache-append`. The ACLNN package, direct library and Python
extension were rebuilt together from this implementation in a clean build,
then installed under a private vendor directory. Documentation updates after
this commit do not change the tested implementation.

## Environment and scope

| Item | Development environment |
| --- | --- |
| Device | Ascend910_9382, one logical device, 48 Vector / 24 Cube cores |
| CANN | 9.1.0 |
| Python | 3.12.13 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4, git `5dd8ef3f9b375b5ae4a83538d5785754148c3302` |
| NumPy / ml_dtypes | 2.2.6 / 0.6.0 |
| OS | Ubuntu 22.04.5; task specifies openEuler 24.03 LTS-SP3 |
| Driver | 26.1.1; task specifies 25.5.t6.b001 |

These are development-device results. The OS and driver differences remain
unresolved for formal acceptance. The full xLLM inference server was not built;
the new native C++ wrapper passed syntax/type compilation against the real
xLLM headers and its `C10_USE_GLOG`, `TORCH_HIGHER_THAN_PTA6` and
`TORCH_SETCUSTOMHANDLER=ON` definitions.

## Executed checks

| Check | Result |
| --- | --- |
| Unmodified public verifier | 35/35 passed; zero raw-bit mismatches |
| Extended suite | 206/206 passed |
| C++ two-stage ACLNN sample | FP16 and BF16 bit-exact; workspace 0 |
| Host task mapping | 80 shape/core/offset combinations; 321768 tasks checked |
| Long host queue | 8192 calls per entry; allocations return to baseline |
| Python storage release | 100 cross-stream pressure rounds per entry passed |
| Large odd-row boundary | 8/8 passed: both entries, FP16/BF16, offsets 0/1 |
| Official performance and checker | 10/10 passed; both processes exit 0 |

The extended suite includes 70 public-case records across both entries,
24 extra-input records, 84 invalid-input records, metadata/verifier checks,
two-stream prepared calls, early input/output release, repeated append,
20-call allocation recovery and ten memory cases. Memory checks inspect both
allocated and active bytes; all peak increments remain within output bytes
plus 16 KiB, with zero explicit workspace and return to the warmed baseline.

The additional boundary shape is B=4, H=128, L=1024, D=65. Its K cache prefix
ends after 359 of 512 rows, at byte offset modulo 32 equal to 14. The check uses
all raw uint16 patterns and verifies input preservation and independent output
storage. The direct build shares mapping/copy semantics but does not enable
the legacy compiler's unsupported L2-hint runtime configuration.

## Official end-to-end performance

The original `benchmark.py` and `check_performance.py` were used unchanged:
50 warmups, 200 calls per round, five rounds, median of round means. The public
`run` entry allocates fresh outputs and executes both ACLNN stages; allocation,
host submission and completion are included. These values are wall time in
microseconds, not isolated kernel timings.

| Official case | Measured (us) | Limit (us) |
| --- | ---: | ---: |
| T5SplitQkvAppendCache_001_float16_random | 18.59530 | 28 |
| T5SplitQkvAppendCache_005_float16_random | 18.30965 | 28 |
| T5SplitQkvAppendCache_006_float16_random | 19.01450 | 29 |
| T5SplitQkvAppendCache_007_float16_random | 74.29595 | 75 |
| T5SplitQkvAppendCache_008_float16_random | 18.60170 | 29 |
| T5SplitQkvAppendCache_011_bfloat16_random | 17.74520 | 28 |
| T5SplitQkvAppendCache_015_bfloat16_random | 18.14985 | 28 |
| T5SplitQkvAppendCache_016_bfloat16_random | 18.63915 | 29 |
| T5SplitQkvAppendCache_017_bfloat16_random | 75.40325 | 77 |
| T5SplitQkvAppendCache_018_bfloat16_random | 18.96190 | 29 |

The large-case margins are small. Retest the fixed protocol after changes to
the implementation, toolchain, driver, hardware or allocation environment.
Earlier failed and interrupted attempts are preserved separately; they do not
contribute samples to this table. An earlier experiment accidentally omitted
a cache-policy call; its passing measurement was not used to validate the
budgeted implementation. The final source uses explicit normal/bypass aliases
and passed the full checks after a clean rebuild.

## Evidence and reproduction

Run the commands in the project README against the original external task
package. The task package's `case.json` SHA256 is
`9ce34ba31fc08abe1969ad4997f7b6173fa950533473d6977398a8ed4db741e8`.
Its task-book checksum discrepancy and conflicting relative/absolute timing
language remain recorded; no supplied acceptance files were changed.

The frozen development archive is `evidence_ab70af7.tar.gz`, SHA256
`cc9d89474fdefe205035b26cebdc4d2174f37692a2cb31a0cf92e06e405dcfaa`.
It contains actual stdout/stderr, per-step exit codes, all round timings,
source hashes, pre-test artifact hashes, loaded-library identity, the private
`.run` package, direct library and Python extension. Artifact hashes were
checked again after testing. Binaries and environment-specific logs are kept
outside this source repository in the task's development evidence directory.

The clean package build used `T5QKV_BUILD_DIR=build_final_176`, followed by
`scripts/install.sh` into a private prefix and `scripts/build_framework.sh`.
No global CANN installation or unrelated server project was modified.
