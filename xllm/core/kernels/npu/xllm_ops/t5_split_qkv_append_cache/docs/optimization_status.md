<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Current optimization result (2026-10-30)

The final kernel at `8ea9affaf131b22c8fb786b79da5968ce6724744` passes all ten
absolute limits in five independent fresh-process batches on the current server.
The latest native-terminal paired batch measures 007 at 72.10475 us (limit 75)
and 017 at 72.16775 us (limit 77). The full regression passes 35 public accuracy
cases, 206 extended records, queue pressure, cache/tail/hot geometry boundaries,
and the 104-check / 171584-task host mapping suite. The reference passes all 35
accuracy/storage cases and runs with the same allocator and public timer.

This result requires `PYTORCH_NPU_ALLOC_CONF=page_size:1g` before importing torch,
for both custom and baseline processes. Normal/expandable allocator trials were
unstable and remain diagnostic failures. The new manual event pipeline keeps
one task ahead across cache classes, uses two 92 KiB slots, and admits a bounded
Value-input prefix while retaining Key input/output. For 007/017 this normal
logical working set is 188.15625 MiB under a 190 MiB software admission budget.
The budget is not a physical residency guarantee. The 1 GiB large allocator pool
and 2 MiB small pool are measured process reservations, separate from zero
explicit operator workspace, 184 KiB UB and active output allocation.

The reports and submission now promote this passing current-environment result,
preserving workbook styles and all 35 case rows, with three fresh real terminal
captures and immutable original logs. See `task_submission/` at the repository
root for the complete design, reproduction steps, reports and evidence index.
The original server archive SHA-256 is
`2f142c1f953965ef4353df283c03086c561abdb2aa7ff3ef8e5f269a6af8bda1`.

Formal acceptance still has explicit environment, taskbook checksum/criterion,
TBE-reference and full-service/community-process gaps. The user authorized this
Ubuntu22.04.5 / driver26.1.1 environment; no global policy or clock was changed.

## Historical checkpoint (superseded by the result above)

The following text and its evidence describe prior failed candidates. They are
retained for traceability and do not supply the current reported measurements.

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

`reports/visible_phase172_20261030_performance.json` is the latest retained
comparison, executed live in the SSH terminal for native screenshots. The custom and the unique task-book `views_concat_q_copy` reference
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
| 001 float16 | 17.093850 | 28 | 112.573700 | 0.151846 | passed |
| 005 float16 | 17.282500 | 28 | 119.788150 | 0.144276 | passed |
| 006 float16 | 18.910650 | 29 | 116.062850 | 0.162935 | passed |
| 007 float16 | 93.429350 | 75 | 233.842400 | 0.399540 | failed |
| 008 float16 | 18.217550 | 29 | 93.914500 | 0.193980 | passed |
| 011 bfloat16 | 17.272850 | 28 | 82.234050 | 0.210045 | passed |
| 015 bfloat16 | 17.755550 | 28 | 93.641200 | 0.189613 | passed |
| 016 bfloat16 | 18.173200 | 29 | 87.863850 | 0.206834 | passed |
| 017 bfloat16 | 92.646550 | 77 | 233.942100 | 0.396023 | failed |
| 018 bfloat16 | 18.467650 | 29 | 86.514850 | 0.213462 | passed |

The custom and reference child exit codes are both 1. The aggregate status and
its exit code remain failed because the custom exceeds two limits. The comparison
helper now writes complete diagnostic rows even on absolute-limit failure,
instead of dropping the aggregate. The earlier `checkpoint172_paired_final` run has identical pre/post source and
artifact manifests. The subsequent native-capture run retains the kernel/API
hashes in its display output; it is a separate run and does not inherit that
earlier manifest-freeze check.

## Rejected candidates and repeatability

| Rejected/checkpoint run | 007 us | 017 us |
| --- | ---: | ---: |
| `optimized_cache_phase172_default_final_01` | 81.3938 | 81.7986 |
| `optimized_aligned_cache_phase172_final_01` | 104.99625 | 104.34265 |
| `optimized_stripe176_final_01` | 83.9896 | 83.73925 |
| `optimized_stripe176_stagger_final_01` | 87.10885 | 86.80975 |

The unchanged checkpoint's earlier 81-us and 88-us measurements and the latest
93-us native-capture measurements are all retained. Earlier candidates, including a four-batch
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
clocks were changed. The user explicitly authorized continuing on this environment on 2026-10-30;
matching the acceptance image is not a blocker for continued development.
The environment mismatch remains disclosed for formal acceptance.

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
for updating the official reports/package has not been met. After the user unlocked the Mac, three native terminal screenshots were captured
for newly executed precision, memory and paired performance checks; see below. This is a
reviewable development checkpoint, not ready-to-submit material or formal acceptance.


## Native screenshots and new hardware counters

The user unlocked the Mac on 2026-10-30. The original, unedited 2880 x 1800 JPEG
captures are `phase172_accuracy_native.jpg`, `phase172_memory_native.jpg` and
`phase172_performance_native.jpg` under `evidence/20261030/`. They show a fresh
35-case precision pass, ten memory measurements with release recovery, and an
8/10 paired performance result respectively. The screenshots show a CLI view
explicitly derived from that actual execution's JSON; they are not original
benchmark stdout. The original child stdout, JSON, reference qualification,
five-round samples, display CLI and display stdout are retained separately.
No new screenshot was inserted into the old XLSX reports.

`native_execution_evidence_index.json` links each capture to those files and
its kernel/API hashes. `visible_phase172_20261030_evidence.tar.gz` contains 182
raw files, verified individually and against server archive SHA-256
`5e3f2261542e19e3bfd5b25a0a6fcc92cfb2862afc014184281ce906f0d87b01`.

Two additional `msprof` diagnostics collect PipeUtilization and Memory separately.
Each executes 80 calls per large case; the derived counter summary discards the
first 64 and reports the final 16. In PipeUtilization, 007/017 task-duration
medians are 84.692/85.542 us. MTE2 active-ratio medians are 0.8385/0.8355,
MTE3 0.828/0.819, and scalar 0.103/0.102. These counters overlap in time and must
not be added together. They support investigating memory movement and overlap
before scalar-only changes. The separate Memory capture reports L2 bandwidth
as zero; this is insufficient to infer cache-hit rate or zero L2 traffic.
Profiling overhead and the changed workload make all these timings diagnostic,
not substitutes for the official 50/200/5 measurements.

A targeted triple-buffer trial (`optimized_triple60_final_01`) used three 60 KiB
UB slots, a 30,720-element tile and two-task lookahead. It passed public precision,
206 extended records, queue/cache/tail checks and 104 host combinations
(191472 decoded tasks), but 007/017 remained 92.41635/92.4474 us. The trial is
rejected; the retained kernel and screenshots remain the two-buffer checkpoint.


The second counter-driven candidate, `optimized_balanced172_final_01`, divides
172 MiB admission between the same row prefix of both K/V history inputs and
both outputs. It passed the functional gates but measured 92.41095/91.98115 us
for 007/017. It was also rejected. Restored kernel sources match the committed
two-buffer policy. A direct-entry diagnostic using the unchanged public timer
measured 214.48345/214.02325 us; it shows no shortcut via entry substitution.
The task book explicitly requires `run` to call ACLNN, so the mandatory entry
remains unchanged. Runtime/cache behavior may differ between the two launch
paths; this comparison alone does not isolate pure API overhead.

`native_trials_20261030_evidence.tar.gz` retains the two failed candidates,
their raw build/test logs, installers, installed files and reconstructed tested
sources, plus the independent direct-entry diagnostic. SHA-256:
`bf57567d7d4d92895c0b7f6384852de90fb03d2c25ad40917aedb1e170578c17`.
All 144 archived files were checked locally against its manifest. Each candidate
has 23 reconstructed source/build/test files matching its pretest manifest,
including the actual clang-format output. Both runs stop on performance failure
before producing a posttest manifest; they are not passing acceptance freezes.
The archive manifest and `DERIVED_TRIAL_SUMMARY.json` preserve that limitation.
