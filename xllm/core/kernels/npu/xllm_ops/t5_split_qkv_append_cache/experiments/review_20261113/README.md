<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Review remediation candidates (2026-11-13)

These are **unvalidated A3 performance experiments**, not a new accepted product
implementation. The default product source and existing XLSX/log/ZIP bytes are
unchanged. The reviewer still reports case 007 over 75 us; case 017 must also be
retested against 77 us. See `docs/review_feedback_20261113.md` and the remediation
plan in this directory. No speedup follows from the host-only checks below.

## Single-variable source candidates

`prepare.py` reads product sources and all 12 original task files from the fixed
Git commit in `candidates.json`. It does not copy dirty/untracked files or old
binaries. Each variant has its own project/install/build paths and a hash
manifest. The control is byte-identical to the pinned sources. C1/C2 change only
`kReadBalanceBudgetBytes` in the shared host/direct/device header, preserving
52-byte tiling, the 184 KiB double buffer, manual events, tail fusion and cache
hints. There is no runtime environment switch that can accidentally affect one
build path while leaving another unchanged.

| Candidate | Admission budget | Normal Value rows at B16 H16 L2048 D64 | Normal logical bytes |
| --- | ---: | ---: | ---: |
| control | 190 MiB | 240 | 197296128 |
| c1_186mib | 186 MiB | 224 | 193101824 |
| c2_182mib | 182 MiB | 208 | 188907520 |

The budget applies generically through the existing geometry policy, not a case
name or dtype check. Smaller admission can increase bypass reads and make timing
worse. Other geometries therefore require the full boundary regression too.
Constant-geometry/scalar and event-order changes remain gated on fresh profiling.

## Offline preparation

Run from a clean checkout containing the pinned commit. Set these paths for the
current host; Python 3.12 and a C++17 host compiler are sufficient here.

```bash
export REPO=/absolute/path/to/xllm
export EXP="$REPO/xllm/core/kernels/npu/xllm_ops/t5_split_qkv_append_cache/experiments/review_20261113"
# Local candidates belong under the workspace .codex-worktrees/t5qkv/ directory.
export PREPARED=/absolute/path/to/cann/.codex-worktrees/t5qkv/new-review-run
python "$EXP/prepare.py" --repo "$REPO" --output-root "$PREPARED"
python "$EXP/check_offline.py" --prepared-root "$PREPARED" --report-dir /absolute/path/to/new-offline-evidence
```

Existing output directories are rejected. Preserve partial/failed runs and choose
a new directory. The host checker compiles the actual generated candidate
header, checks independent expected row/byte counts, then runs the existing
104-shape/core/offset, 171584-task CPU address mapper for each variant. It does
not compile Ascend C kernels or emulate the manual-event DMA pipeline.

## Build on the target toolchain

On a server, use a fresh `/workspace/t5qkv-review-<run>/` directory and prepare
the same pinned sources there. Source the CANN environment, activate the
Python/torch_npu environment, and run `build.py` once per candidate. Example:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Activate the Python environment matching the task before continuing.
python "$EXP/build.py" --prepared-root "$PREPARED" --candidate control
python "$EXP/build.py" --prepared-root "$PREPARED" --candidate c1_186mib
python "$EXP/build.py" --prepared-root "$PREPARED" --candidate c2_182mib
```

The original SDK, private installer and forced framework/direct build scripts are
used. Fresh trees are required; build failures retain original stdout/stderr and
argv/cwd/environment/exit records. Each successful build records all installed
files, the SDK package, direct library and extension hashes. `built_a3_unvalidated`
means compilation only and must not be reported as precision/performance passed.

## Later A3 measurement

`run_matrix.py --plan-only` prepares the process schedule without importing
torch or touching an NPU. Actual execution first verifies the candidate source,
build artifacts and original task-package hashes, then probes both allocator
contexts. Anything other than exactly one visible `Ascend910_9382` is rejected
before timing. This is a chip guard, not proof of a fully matching formal OS and
driver; runtime details are retained and differences remain acceptance gaps.

```bash
python "$EXP/run_matrix.py" --prepared-root "$PREPARED" --report-dir /absolute/path/to/new-plan --plan-only
python "$EXP/run_matrix.py" --prepared-root "$PREPARED" --report-dir /absolute/path/to/new-screen
# After selecting a candidate, use previously unused confirmation processes:
python "$EXP/run_matrix.py" --prepared-root "$PREPARED" --report-dir /absolute/path/to/new-confirm --phase confirm --candidates c1_186mib
```

Screening defaults to three fresh processes per candidate/allocator. Candidate
and allocator orders rotate/reverse by batch. Confirmation requires one candidate
and at least ten batches **per allocator**. Default allocator clears inherited
`PYTORCH_NPU_ALLOC_CONF`; 1g sets exactly `page_size:1g` before each child imports
torch. Both contexts clear inherited queue/blocking overrides and select logical
device zero. An exclusive task-directory lock prevents overlapping runs of this
driver; independently check for other users' NPU workloads before launching.

Each process executes the unchanged complete ten-case public `benchmark.py` with
the original 50/200/5 timer and newly allocated outputs. No case filtering,
extra cache conditioning, sample dropping or automatic retry is introduced.
Reference accuracy is qualified on all 35 cases per allocator; its own absolute
threshold failures are retained as baseline timing, not custom failures. Custom
threshold failures remain in the matrix and force a failed aggregate while the
remaining planned custom batches continue. Incomplete/invalid evidence is also
failure. Round means/medians and the embedded original check results are
independently reconciled without rewriting the original JSON or stdout.

The driver freezes source, scripts, package and build artifacts before/after;
mid-run identity changes invalidate the result. Each child has its own original
log/JSON and execution record. A source/artifact freeze plus explicit library
selection is not a complete loaded-library-map audit; final runtime identity,
profiling, memory and full functional regression still have to be collected.

Even a confirmation summary with `status=passed` is **performance-only**. Complete
35-case precision, 206-record engineering, ten memory checks, queue/lifetime,
cache/tail/hot-geometry, two-entry and C++-example regressions, same-device ratio
reporting and new genuine screenshots are required before updating delivery.
The published thresholds and XLSX formats are unchanged.
