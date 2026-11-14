<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Same-device binary/address investigation

Target: Ascend910_9382 on the restored A3. Existing source, binaries, official
cases, timing protocol and XLSX files remain unchanged.

`audit_binary.py` checks old artifacts against historical 20261030 SHA values,
new artifacts against their frozen build manifest, and both source inventories.
The retained result shows identical product sources, whole SDK kernel objects,
opapi and direct library. The extension .text is equal, but .init, dynamic-link,
build-id and debug/symbol sections differ. Do not call the entire extension equal.

`run_crossover.py` runs old/new artifacts with each visible chip and default/1g
allocators. Each combination passes the full 35-case verifier before timing;
three rotated/reversed fresh benchmark processes then execute the complete
original ten cases with the original 50/200/5 protocol. Threshold failures stay
in the results; a completed diagnostic investigation is not an acceptance pass.
`after_benchmark.py` performs identity work only after the original script exits:
loaded library hashes, allocator pool snapshot, PID and npu-smi physical mapping.
It adds no hook inside the timing loop. All frozen files are checked before/after.

`memory_history.py` is separate instrumentation. It records actual alloc/free
history with stack recording disabled while running the original full benchmark.
Its times are diagnostic only. These addresses are device virtual addresses and
allocator segments, not exposed physical HBM addresses or proven cache colors.
A matched L2 profiler run is intended to correlate counts with allocations from
the same process; different metric/process samples must not be spliced together.

Remote investigation root: /workspace/t5qkv-identity-20261114-01. The crossover
is complete: 8 precision processes passed all 35 cases and 24 timing processes
completed with unchanged source/artifacts. Both old/new versions fail default
allocator limits on both chips, while page1g is faster. No new performance pass
is claimed.


The complete 31556 allocation sizes match the full public workload exactly in
all four memory-history runs; each large case uses one stable output address
triple for all 1000 timed calls. PID/driver tables resolve visibility 0/1 to NPU3
chips 0/1, physical IDs 6/7 and PCIe 91:00.0/93:00.0. Default allocation VAs are
identical across these two chips, but same-run 007 kernel medians are about
124/95 us and r0 read-miss-allocation events about 242k/140k. Page1g medians are
about 72 us and event counts about 0.5k/1.7k. These raw events are not universal
cache hit percentages. The observed bottleneck is cache residency/access
sensitivity associated with segment layout and physical-device mapping;
physical HBM page/cache-index addresses remain unavailable.

Default large input and output base VAs all have zero low 21 bits. Page1g's
input/output data bases differ by small offsets (1024,1536,2048,35328 mod 2MiB).
C6/C7 therefore test placement without changing the kernel: both allocate a
fresh independent Key storage with 2KiB slack when the existing cache-admission
policy is active. C6 places logical Key data after that prefix; C7 starts at the
original base and leaves identical slack at the tail. Output shapes, continuity,
ownership and current stream remain unchanged; no input copy/global allocator
setting/output reuse/explicit workspace is introduced. The task permits
independent contiguous tensors and a 16KiB memory-overhead allowance; actual
memory and release checks must still pass. The derived eight-case cache boundary
test asserts the real Key storage offset and records data/base pointers.

The historical padded-allocation failures do not qualify this new controlled
experiment. C6/C7 are unvalidated hypotheses, and a gain only in one allocator
or one fresh process does not qualify either candidate.

Retained archives (individually SHA-verified): crossover 139 files,
560f3e544a17ba304bca98d1994f01cd38611f26f5dbfd2bad5d9c7e0f76cd68;
address/PMU 86 files,
cfa8b6d204d31e049d058beef289150da50005f3f784efd257d1d15b16fc920e.


## Distributed Value-input candidate (C8)

C6/C7 did not pass default allocator timing. C8 changes neither output allocation
nor the 190MiB admission budget. At C=3, 16 history tasks/core and 15 admitted
Value tasks/core, it maps each core's normal Value phase to the first 15 tiles
of that core's corresponding 16-tile Key interval; the bypass phase copies its
last tile. Only phase entry changes. Existing incremental cursors, buffer events,
append fusion and all other geometries retain their behavior. 48 bypass tiles
contain 16 of each tile position, so admitted/bypassed read bytes remain exactly
60MiB/4MiB. No extra hot-loop branch is added.

This is distributed **Value input** admission, distinct from historical rejected
172/176MiB striped **Key output** admission. The hypothesis is that removing the
small quota uniformly, and aligning K/V core affinity, reduces concentrated
cache pressure; lower same-process read-miss counts/default time are required.
`check_distributed_mapping.cpp` compiles the actual candidate helper and proves
768 tasks covered once, 256 unique append tiles and unchanged byte counts;
this is host-only evidence. A new eight-record NPU case uses row-tagged uint16
patterns across all 256 rows, both dtypes, entries and input offsets. Its metadata
uses equivalent cached-row count because whole rows are no longer the unit of
admission. Existing public tests and timing cases remain unchanged. This candidate
has not passed A3 performance screening and must not replace the product yet.


## Two-plane residency controls

C8 completed its functional checks but failed default timing. The next bounded
comparison removes the third large cached plane rather than shifting addresses
or sweeping a near-capacity budget:

| Candidate | Normally cached history | Normally cached large outputs |
| --- | --- | --- |
| C9 input-only | Key and Value | neither |
| C10 Key chain | Key | Key |
| C14 output-only | neither | Key and Value |

At the large public geometry, normal logical working bytes including QKV/Query
are approximately 128.125/128.156/128.188 MiB. Hints are not residency guarantees.
All plans still copy every output, allocate fresh independent tensors, and retain
52-byte tiling/184KiB UB/manual events. Small shapes keep the original policy;
C10/C14 retain the working-set guard and original fallback outside their domain.
C14's legacy host quota-helper fields do not describe its overridden input/output
hints; its actual_normal_* fields document the intended output-only plan. An
initial CPU check caught a missing large-shape budget guard before any target
build; the failing preparation/logs were retained and the guard corrected.

C9 trades cached outputs for normally cached reads; C10/C14 compare cache roles
at similar total working size. Extra HBM traffic can outweigh any residency gain.
The additional eight-case fixture explicitly checks each cache plan, not a
nonexistent partial-admission boundary. None is a performance-qualified version
until full real pre-screen and independent original-protocol timing finish.


All three two-plane plans completed 24 functional children and 24 full timing
processes but failed absolute limits: input-only/output-only are about 94–96 us,
Key-chain-only about 125 us. Reduced residency has not overcome its measured
extra memory-transfer cost. C15 therefore preserves the original near-188MiB
cache plan and kernel, changing only large-output allocation order (Value before
Key, returned order still Query/Key/Value). It uses the same three independent
allocations and sizes, no slack, dummy allocation, pool priming, global allocator
option or extra workspace. Small shapes retain their original order. Any gain
must survive complete independent-process tests; output addresses are recorded
rather than assuming the allocator follows a particular order. The new screen
will also use --post-identity to verify actually mapped libraries per process.
