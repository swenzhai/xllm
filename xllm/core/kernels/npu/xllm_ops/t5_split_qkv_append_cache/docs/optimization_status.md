<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Optimization checkpoint (2026-10-30)

Performance acceptance remains incomplete: the final paired run passes 8/10
absolute limits. Of the original five failures, 001, 005 and 018 are below their
limits; 007 and 017 remain above 75/77 us. No passing delivery is promoted.

The retained kernel implementation is commit `a221c7c` on
`t5-qkv-task-submission`: a 172 MiB software cache-admission budget, separate
contiguous cached-K, bypassed-K and V phases, 47,104-element tiles, two 92 KiB UB
buffers, one-task-ahead prefetch and guarded history/append tail fusion. Host
tiling selects fusion once; the device avoids repeating its geometric division.
The framework build forces recompilation after header changes. Outputs retain
separate allocations. Aligned basic DataCopy, padded allocations and striped
cache admission (including alternating per-core phase orders) were rejected.

## Latest same-device measurement

`reports/checkpoint172_paired_final/comparison.json` is the latest retained
comparison. The custom and the unique task-book `views_concat_q_copy` reference
ran in independent processes on logical device 0, using the unchanged official
fixture generator, verifier and 50-warmup / 200-call / five-round benchmark.
The reference passed all 35 public accuracy cases; both timed implementations
preserved all bits in their ten performance cases. Identical generated input
values are checked by per-case hashes; device addresses are independent.
The table displays microseconds rounded to six decimals. Original five wall-time
and device-event samples remain in the raw JSON. Ratios are dimensionless measured
custom/reference values, not an interpretation of the task-book's ambiguous ratio
requirement. Reference absolute-limit failures are retained as measurements;
they do not qualify or disqualify the custom implementation.

| Case | Custom median (us) | Limit (us) | Same-device baseline (us) | Custom / baseline | Absolute check |
| --- | ---: | ---: | ---: | ---: | --- |
| 001 float16 | 16.1984 | 28 | 83.928 | 0.193004 | passed |
| 005 float16 | 16.86915 | 28 | 94.6977 | 0.178137 | passed |
| 006 float16 | 16.6323 | 29 | 85.5223 | 0.194479 | passed |
| 007 float16 | 87.9758 | 75 | 226.4281 | 0.388537 | failed |
| 008 float16 | 16.9616 | 29 | 95.14285 | 0.178275 | passed |
| 011 bfloat16 | 16.321 | 28 | 84.29195 | 0.193625 | passed |
| 015 bfloat16 | 16.89525 | 28 | 94.40745 | 0.178961 | passed |
| 016 bfloat16 | 17.14895 | 29 | 84.6871 | 0.202498 | passed |
| 017 bfloat16 | 87.98345 | 77 | 217.43475 | 0.404643 | failed |
| 018 bfloat16 | 16.99735 | 29 | 96.8451 | 0.175511 | passed |

The custom and reference child exit codes are both 1. The aggregate status and
its exit code remain failed because the custom exceeds two limits. The comparison
helper now writes complete diagnostic rows even on absolute-limit failure,
instead of dropping the aggregate. The pre/post source and artifact manifests
are identical. The final restored checkpoint was force-recompiled before this
comparison; its shared/kernel source matches the retained implementation.

## Rejected candidates and repeatability

| Rejected/checkpoint run | 007 us | 017 us |
| --- | ---: | ---: |
| `optimized_cache_phase172_default_final_01` | 81.3938 | 81.7986 |
| `optimized_aligned_cache_phase172_final_01` | 104.99625 | 104.34265 |
| `optimized_stripe176_final_01` | 83.9896 | 83.73925 |
| `optimized_stripe176_stagger_final_01` | 87.10885 | 86.80975 |

The unchanged checkpoint's earlier 81-us measurements and the latest 88-us
measurements are both retained. Earlier candidates, including a four-batch
passing experiment, later failed repeat measurements. None establishes stable
closure of 007/017. The table above uses the latest paired batch, not the fastest
historical batch. Striped cache admission and alternating K/V order were restored
to the committed contiguous policy after their regressions.

The full contiguous-policy checkpoint run
`optimized_cache_phase172_default_final_01` passed 35 public precision cases,
206 extended records, queue pressure (8192 calls and 100 release rounds per
entry), eight cache-boundary checks, 16 tail-boundary checks and 104 host
shape/core/offset combinations (171584 decoded tasks), before performance failed.
The two striped candidates passed the same functional checks before their
performance failures; they are not the retained implementation. The final paired
run separately qualifies the reference and checks the ten timed custom cases.
Memory measurements in the extended suite cover one call and release recovery
per performance case; 20-call recovery is recorded only for case 018 in both
entries. Explicit workspace remains zero.

## Analysis and remaining work

Profiling the unchanged public benchmark recorded 10510 kernel executions.
For the profiled flat-cursor diagnostic, post-warmup kernel medians for 007/017
were approximately 81.61/81.32 us, close to end-to-end 82.76/81.89 us. Its trace
and original profiler CSV are retained. This points to device memory movement
for that slow run; it does not prove a cache-hit rate or explain the run-to-run
variation. Output addresses were stable within an instrumented repeated-call
diagnostic. Loaded API/tiling library paths were checked, and a separate no-write
canary produced its expected precision failure, confirming the selected kernel
is loaded. Canary failures are diagnostics, not official acceptance records.

The process exposes only logical device 0. Python 3.12.13, torch 2.10.0+cpu,
torch_npu 2.10.0.post4 (git 5dd8ef3f9b375b5ae4a83538d5785754148c3302)
and CANN 9.1.0 match the requested versions. Ubuntu 22.04.5 and driver 26.1.1
differ from the required openEuler 24.03 image and driver 25.5.t6.b001. This is
an acceptance/environment gap, not a proven cause of the slowdown. Read-only
checks reported device-share=False and UNION_POLICY. No hardware policy or
clocks were changed. Further tuning needs measured MTE/cache counters for the
slow run, or repeat verification on the specified image; current evidence does
not justify another speculative cache-policy change.

## Frozen diagnostics and delivery boundary

The complete raw archive is
`evidence/20261030/optimization_20261030_diagnostics.tar.gz`.
SHA-256: `eaa34ee731d81f45c1a4c0a268946321ae77f2a8e2695f8c3a846827d1736631`.
It contains 860 files with a per-file `raw/SERVER_MANIFEST.json`: original logs,
JSON, profiler CSV, tested source snapshots, restored source, installed package
and libraries, benchmark drivers and environment evidence. The archive SHA was
verified against the server and every archived manifest entry was recomputed
locally. `reports/archive_optimization_20261030_metadata.json` was an empty
redirect target while the archive was produced; the completed archive metadata
is provided separately beside this document's evidence. It is not a test log.
The earlier checkpoint-only archive remains available alongside it.

The repository and workspace `task_submission` remain the previous frozen
35-file delivery at commit `5c6db23d43b4b26e5d58430420babf0c69eba91c`.
The existing XLSX files and installer do not represent this optimized checkpoint.
All three XLSX SHA-256 hashes match the pre-optimization backup and repository
copies, as recorded in `xlsx_optimization_unchanged_audit.json`. Their content,
format and screenshots were not changed. The complete-pass condition requested
for updating the official reports/package has not been met. Native screenshots
for new tests also remain unavailable while the Mac is locked. This is a
reviewable development checkpoint, not ready-to-submit material or formal acceptance.
