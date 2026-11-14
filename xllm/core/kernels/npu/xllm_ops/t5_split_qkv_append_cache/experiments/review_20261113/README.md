<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# Review remediation candidates (2026-11-13)

These are **isolated A3 performance experiments**, not a new accepted product
implementation. The default product source and existing XLSX/log/ZIP bytes are
unchanged. The reviewer still reports case 007 over 75 us; case 017 must also be
retested against 77 us. See `docs/review_feedback_20261113.md` and the remediation
plan in this directory. No speedup follows from the host-only checks below.

## Isolated source candidates

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
C1/C2 were rejected after the first real A3 screen: neither improved the large
cases. The control passed three page1g batches, but failed all default batches.
Profiling showed much higher MTE2 time and read-miss allocation counts with the
default allocator; scalar time was about 4.9 us in both contexts. Instrumented
times are diagnostic, not acceptance samples.

C3/C4 instead keep all historical inputs normally cached, bypass Value output,
and admit a prefix of Key output. They use the existing streaming scheduler and
preserve the 52-byte tiling and event pipeline. At B16 H16 L2048 D64, C3 admits
224 Key rows (193097728 normal logical bytes), C4 admits 240 (197294080 bytes).
The prefix is rounded down to a common multiple of the core-balance group and
the 512-byte output-boundary group. Both candidates are unvalidated performance
hypotheses until measured. Their trusted patches touch only the policy header.

Offline checks cover all five candidates. The new boundary regressions exercise
both partial admission (L1023 D65; 432/456 Key rows) and zero admission (L1024
D65; group too large). The latter deliberately checks the no-admission edge.
These tests change their own derived inputs/golden consistently; the original
35 public cases and ten public timing cases remain unchanged.

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

After building, run a functional pre-screen before timing:

```bash
python "$EXP/run_functional.py" --prepared-root "$PREPARED" --report-dir /absolute/path/to/new-functional
```

This checks 35 public accuracy cases, 206 engineering records and eight cache
boundary records for each candidate and allocator. `--full` also includes queue,
tail-fusion and hot-geometry tests. The pinned legacy boundary test assumed a
190 MiB budget in its metadata calculation; the driver archives a separate
executed copy with that expected-budget literal adjusted per candidate. Product
and built test trees remain unchanged, and both original and executed test
hashes are retained. This adjustment changes no inputs, golden or pass criteria.

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

## Restored A3 follow-up

Use explicit `--candidates control c3_input_priority_186mib c4_input_priority_190mib`
for both functional screening and the timing matrix of the second isolated run.
Build those same three names separately first. `run_functional.py` adds eight
zero-admission records per allocator for each input-priority candidate, in
addition to its eight partial-boundary records. This makes 22 functional child
processes for the selected three candidates across both allocator contexts.

`collect_profile.py` collects the unchanged full benchmark under PipeUtilization
and L2Cache, separately per allocator, preserving failures and source/artifact
identity. `profile_application.py` explicitly adds the selected Python's site
packages because the profiler launch otherwise omitted them; an initial failed
launch is retained. `summarize_profile.py` accepts only the complete four-job
collection and selects 1051 T5SplitQkvAppendCache tasks per public case, retaining
the last 1000 after accuracy/warmup. Counts and instrumented timings are not
acceptance results or universal hit-rate percentages.

First-run raw evidence is retained on the A3 under
`/workspace/t5qkv-review-20261113-01`; compact archives and derived summaries are
under the workspace `development_evidence/20261114_*`. C1/C2 rejection and
profiling do not qualify the default product or the historical delivery package.

## C5 address-role experiment

The second 18-process screen rejected C3/C4 for default-allocator timing. C4
large cases were around 83 us in all three default batches versus about 70 us
with page1g. Its instrumented default runs also varied between about 82 and
117 us; read-miss allocation counts remained much larger than page1g. Five small
case checks failed in its first default/page1g batches and remain in the evidence.
Neither candidate is promoted.

C5 mirrors the original control's balanced cache roles without changing its
190 MiB budget or phase/event schedule: cache all Value input/output, normally
cache the 240-row Key-input prefix and bypass Key output. This tests whether
which physical allocations occupy the cache matters, rather than scanning more
budgets. The fallback path and shared quota helper remain unchanged. The
`normal_key_rows`/`normal_value_rows` configuration and host checker fields for
C5 describe that unchanged **legacy helper**, not the mirrored kernel roles;
`actual_normal_*` fields state the experimental kernel's intended roles. Host
mapper success does not exercise its new prefetch selection. NPU precision and
raw-bit cache-boundary tests (now labelled Key-input) are required before timing.
C5 is not a verified implementation until those tests and independent timing pass.

## Completed stage result

C5's 12 functional children passed with unchanged source/artifacts. Its 12-process
screen completed: all three page1g batches passed, but all three default batches
failed (007 about 94.5–96.0 us, 017 about 94.6–131.3 us; one default batch also
failed 001/005). C5 is rejected. None of C1–C5 is promoted, no final confirmation
or new acceptance delivery was produced. See [A3阶段复测记录](A3阶段复测记录.md)
and `mirror_screen_results.json` for this bounded candidate-screen stage and
its retained evidence. The next diagnostic direction is same-device old/new
machine-code and allocation/device mapping comparisons, not more budget scans.


## Confirmation with loaded-library identity

The updated driver accepts `--post-identity`. Each custom child runs the original
benchmark via the audited `identity/after_benchmark.py` wrapper, then hashes and
checks all five actually mapped operator libraries and captures PID/npu-smi and
allocator state. No measurement hook or extra allocation runs before/inside the
original benchmark. Wrapper/guard source files join the before/after freeze.
The reference stays the unmodified benchmark. This option is intended for a new
confirmation runner directory; do not overwrite runners used by completed or
active experiments. The current cache-pair screen uses its original frozen
runner, without retroactively claiming this additional per-process check.
Eight host driver/guard tests pass; mocked library fixtures are not NPU evidence.


C18 `flat_output_tiles`：以C4的两历史输入正常缓存、240行Key输出正常/16行旁路、Value输出旁路为基础，将大输出改为连续47104元素tile。跨head处最多三段gather（历史尾/本步64元素/下一行历史头），合并成一段MTE3写入。仅256行、L2048/D64、48核且Key/Value均512B对齐启用；其他沿用原C4。无输出padding/额外GM/全局allocator配置，52字节POD、184KiB UB与双槽事件顺序保持。正常/旁路分界和所有写入块起止均512B对齐。host实际helper逐元素核验67141632元素、1427写tile/2448读取piece、每元素恰好一次；这只是CPU映射验证，不能代表A3通过。

本轮停止条件：C18完成目标构建、18项功能预筛及12批完整原10项协议筛选；若默认或1g任一批失败不晋升，不启动相同方向的盲目参数扫描。通过后才扩展20新进程确认和full回归。外部pool conditioning诊断不参与验收判定。

最终R4诊断入口：[默认分配器定位记录](identity/定位记录.md)。实际API跟踪已确认default/外部预留使用HUGE_FIRST(0)、page1g大段使用HUGE1G_ONLY(6)，均成功；相同绝对虚拟地址不是等价物理分配证明。C18完整功能通过但默认性能仍失败，产品/交付包/XLSX未晋升或修改。API插桩代码仅用于诊断，不参与原协议计时。


R5：基于实测ACL policy0/6差异，单独试验四平面准入C19/C20。C19在两个历史输入、两个大输出上各保留184/256行，C20为176/256行。正常驻留分别约184/176MiB，历史读写旁路流量分别约72/80MiB；相较此前两输入全驻留+单输出部分驻留，将缓存余量分散到四个数据面，检验HUGE_FIRST下的局部访问压力假设，不声称已证明特定物理cache hash。

仅256行/L2048/D64/48核/C3且输出Key/Value基址均512B对齐启用，其余走原C0。四段normal/bypass的prefix边界按4行对齐至512B，旁路核分区旋转抵消normal余数，48核每核32任务，1536任务/512追加全部恰好一次。新GlobalTensor只是同一GM的cache hint视图；52字节POD/184KiB UB/双槽事件发布与消费顺序保持，输出分配/接口/数值/工作区均不变。CPU实际helper覆盖字节数、512B边界、任务均衡、退化域及tiling words，不能替代新kernel的NPU验证；actual_normal_*字段表示新hint计划，原normal_*字段仅描述未改动的C0 helper和fallback。

R5首轮预算：新根独立构建control/C19/C20→26功能子检查（含35/206、8项row-tagged四平面边界及4项提供输出guard）→18批原完整10项50/200/5协议、post-identity五库检查。所有批失败原样保留；若初筛通过，再至少每allocator10个新进程确认及full回归/C++示例，之后才更新正式报告/交付件。若失败，先采集相同源码的L2/Pipe诊断，再据瓶颈决定结构变化，不盲扫准入参数、不更换全局allocator。
