<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Optimization checkpoint (2026-10-30)

Performance acceptance is incomplete. The current checkpoint is the shared
172 MiB cache policy with separate contiguous cached-K, bypassed-K and V phases,
47,104-element tiles, two 92 KiB buffers, one-task-ahead prefetch and guarded tail
fusion. Host tiling selects the fused layout once; the device avoids repeating
its geometric division. The framework build now forces recompilation of header
changes. Outputs retain the original separate allocations; address-padding and
basic-DataCopy trials were rejected.

The original public benchmark is unchanged (50 warmups, 200 calls, five rounds).
The complete run `optimized_cache_phase172_default_final_01` measured 8/10
absolute limits passed. Its original five samples and failure status are retained.

| Initially over-limit case | Current median us | Limit us | Current check |
| --- | ---: | ---: | --- |
| 001 float16 | 16.8263 | 28 | passed |
| 005 float16 | 17.26505 | 28 | passed |
| 007 float16 | 81.3938 | 75 | failed |
| 017 bfloat16 | 81.79860000000001 | 77 | failed |
| 018 bfloat16 | 17.666 | 29 | passed |

Values above are taken directly from the retained `performance.json`; this is
a development checkpoint, not a passing acceptance report. Initial candidates
and even a four-batch passing run had later repeat failures; none establishes
stable closure of 007/017. Independent-process reference measurements are supported
by `tests/compare_same_device.py`, but this failed run stopped before its reference
stage. No earlier reference result is substituted for the current run.

The checkpoint passed 35 public precision cases, 206 extended records, queue
pressure (8192 calls and 100 release rounds per entry), eight cache-boundary checks,
16 tail-boundary checks and 104 host shape/core/offset combinations (171584 tasks).
Memory rows measure one call and release recovery per case; 20-call recovery is
recorded only for case 018 in both entries. Explicit workspace remains zero.

The process exposes only logical device 0. Python 3.12.13, torch 2.10.0+cpu,
torch_npu 2.10.0.post4 (git 5dd8ef3f9b375b5ae4a83538d5785754148c3302)
and CANN 9.1.0 match the requested versions. Ubuntu 22.04.5 and driver 26.1.1
differ from the task image/driver. Read-only checks reported device-share=False
and UNION_POLICY; no hardware policy or clocks were changed. Profiling the
unchanged benchmark showed large-case kernel time close to end-to-end time.
Actual API/tiling library paths were checked, and a separate no-write canary
produced its expected precision failure, confirming the selected kernel is loaded.

The checkpoint's unmodified stdout/JSON and pretest source/artifact hashes are
retained in `evidence/20261030/optimization_checkpoint_phase172.tar.gz`
(SHA-256 `0c4f7d0ee27429983062e6ab69273abb08d7770ab2a828e15997151f87f18b5a`),
verified against the server archive. Additional diagnostic trials remain on the server. The existing `task_submission` XLSX and
installer remain the previous frozen delivery; they are not promoted as the
current optimized implementation. New native screenshots also await manual Mac
unlock. The current code is a reviewable development checkpoint, not ready-to-submit
material and not formal acceptance.
