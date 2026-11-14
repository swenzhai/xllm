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
is running; no new optimized implementation or performance pass is claimed.
