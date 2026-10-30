<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Latest validation

The final tested product source is `8ea9affaf131b22c8fb786b79da5968ce6724744`.
Current evidence: 35 official precision cases, 206 extended records and all
queue/cache/tail/hot geometry/host mapping checks pass. All ten absolute
performance limits pass in five fresh processes with `page_size:1g` configured
before torch import; the latest paired run also measures the independently
qualified taskbook baseline under the same configuration. Ten allocator memory
records meet zero-workspace and output-allocation/recovery checks; reserved pool
memory is reported separately, not hidden in the workspace claim.

The current report ledger, real native captures, original logs, source/artifact
freeze checks and full reproduction commands are in repository `task_submission`.
The following earlier validation notes remain historical; their timestamps and
source revisions must not be substituted for the current evidence.

<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Validation procedure and evidence scope

Use the immutable public task package for all 35 accuracy cases and all ten
performance cases. Preserve its files and checksum anomaly as received. The
public timer must remain warmup=50, iterations=200, rounds=5; event times include
submission gaps and are not pure kernel durations.

Every reproduction exposes one logical device using
`ASCEND_RT_VISIBLE_DEVICES=0`, exported before torch import. Default and
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` trials are recorded separately;
all children, including the reference, inherit the same chosen setting. Verify visible_device_count=1
and record the selected logical device, driver, OS and toolkit. Tests do not
change hardware visibility or allocator settings globally.

Build the private ACLNN package and force a fresh direct-kernel/framework build.
Check source bytes against the implementation commit and record loaded extension,
direct library and API library paths/hashes. Freeze original results before
rendering reports. Source and installed artifacts must match pre/post manifests;
build directories and __pycache__ may be excluded, not build*.sh source files.

Required execution includes:

- Public verify.py: 35 cases with input immutability and separate contiguous outputs.
- acceptance.py: 206 records, including 70 cross-entry public checks, extra seeds
  and storage offsets, invalid metadata/attributes, two streams and lifetimes,
  consecutive appends and ten allocator measurements.
- queue_stress.py: 8192 queued calls and 100 release/pressure rounds per entry.
- cache_boundary.py: eight dtype/offset/entry checks at the 172 MiB hint boundary.
- tail_fusion_boundary.py: 16 dtype/offset/entry checks for an exact full history
  tile (fallback append) and a nontrivial fused tail.
- check_task_mapping.py: 104 shape/core/offset combinations checked independently
  against the affine source mapping, complete coverage and disjoint outputs.
- Original benchmark.py and compare_same_device.py: unchanged public timing and
  a qualified taskbook reference, independent processes and identical generated
  inputs. Preserve every original child log, result JSON and exit code.
- C++ ACLNN example: FP16/BF16 bit equality and zero workspace.

The operator uses two 92 KiB UB buffers and zero explicit global workspace. Ten
memory rows each measure one post-warmup call and release recovery. Twenty-call
recovery belongs only to case 018 in both entries; do not assign it to other
rows. Peaks are allocated/active tensor bytes, not whole-device HBM, reserved
segments or an internal runtime leak audit.

Default-allocator candidates and the alternative flat-output cursor experiment
showed process-to-process performance variation. Their failed measurements remain
in the evidence archive. A prior four-batch passing run was followed by failures;
it does not establish the current delivery status. Profiling the unchanged
benchmark confirmed that the large-case slow path is primarily kernel time,
not repeated device allocations or host submission gaps.

The delivery ledger and reports record the actual current run, including all
five samples, same-device reference ratios, source freeze and unresolved gaps.
This procedure alone is not evidence of a pass. The specified OS/driver image,
taskbook ratio-versus-microsecond discrepancy, original task-manifest discrepancy,
full xLLM service integration, TBE applicability and external community submission
must remain explicit when unproven. A passing development run is not formal acceptance.
