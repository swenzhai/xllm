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


R5首轮26功能通过、18批性能完成但C19/C20失败，12组独立Pipe/L2诊断完成，原源码/产物不变。C19默认Pipe的MTE2约72–76µs（C0约56µs）、Scalar约4.7–4.8µs（C0约4.95µs），MTE3约42–44µs（C0约55µs）；1g原协议约83–84µs（C0约72µs）。缩小输入缓存造成的读代价没有换来足够的写收益；Pipe和L2分开进程，不将两个计数组合成同一次测量，均不作为验收时间。完整指标和失败批见four_plane_screen_results.json。

R5第二轮C21/C22保留C4的两历史输入正常缓存、Value输出全旁路，只把Key输出正常写入分散至每16个连续47104元素tile中的前13或12个。与C18不同：不再是前240行正常Key写入；常驻Key写入分别54706176/50561024字节，含两输入+小张量总正常驻留约180.3/176.3MiB，旁路写79577088/83722240字节。输出512B整块写，跨head最多三段gather，不增加GM/UB/POD/事件/输出存储、不设置全局allocator。48核各29或30任务，CPU逐元素检查67141632元素、1426写tile/全部history读取完整覆盖，实际dense起始与增量跳转一致。输出Key非512对齐时预先选择已支持的Key-chain路径；其他非目标域沿用C4。

C21/C22每allocator须先35/206/legacy8/no-admission8/row-tagged8及supplied-output8（两dtype×Key/Value各offset0/1）真实通过，才跑原完整10项3批筛选。首轮control功能已独立完整通过；新轮control重新构建并每批交错计时，不重复计数旧功能。通过才10批/allocator确认、full回归和正式报告/交付更新。两个固定余量用于检验分散写入及读优先，不做同参数无限扫描。新CLI CPU回归实际解析每套候选的supplied-output命令，防止CPU plan重复实参问题再次发生；host检查不宣称NPU成功。


备用C23/C24（仅CPU准备，尚未A3构建/计时）：分别继承C21/C22的数据流及准入计划；SDK和direct共用一个受完整13字段相等保护的入口，将(16,16,2048,64)的52字节POD以字面常量传给原kernel，给编译器提供地址算术常量化机会。其他metadata原路径，Key/Value指针对齐分支仍在原init中，不更改数据/ABI/UB/workspace。CPU验证字面POD与真实make_tiling逐字节相等，13个单字段变更及3组其他合法shape都不误选；两份逐元素tile检查也通过。此时不宣称机器码已缩减或性能提高。仅当C21/C22的实测剩余差距接近已测Scalar约5µs预算时，才考虑对相应备选做目标构建/功能/原协议测试；不以备用代码改变正在执行的冻结源。


R5第二轮30功能子检查完整通过，18批筛选完成且实际加载五库和前后source/artifact一致。C21默认007三批73.920/73.976/83.842µs，017为73.552/73.763/83.999µs；只有前两批通过，不能合并或丢弃第三批。1g三批均通过。C22默认三批均失败，1g第三批007=75.259µs也超限。8组独立Pipe/L2诊断完成；C21默认Pipe采样task约73.4µs、Scalar4.9µs、MTE2约57.9µs、MTE3约59.5µs；另一个L2进程task约84.3µs、read-miss-allocate约71.6k，不能与前一进程的Pipe值拼为同一次测量。此时最坏差距超过Scalar约5µs，故未单独启动备用C23/C24。514原件归档48335674e6791084c7db29d6197b280f83d2ea0e3a19d0251455c72e6f93f967，逐SHA核验完成，完整结果见dispersed_write_screen_results.json。

第三轮只改变分布、不增减驻留预算：C25以C21为父，将每个完整16-tile周期的三个bypass位置从[13,14,15]改为[4,9,14]，最后不足一个周期仍全normal；正常Key写54706176B、bypass写79577088B严格相同。两份历史读取仍全部normal，Value写全部bypass，连续写块仍512B对齐。C26为同一分布加受13字段guard的字面POD入口，作为独立A/B项测量是否有Scalar余量；尚未声称编译器已缩减机器码。CPU实际helper全67141632元素/1426tile/29–30任务每核覆盖，dense和增量映射一致、目标normal/bypass字节与C21精确相等；C26另通过13单字段变化/3其他合法形状guard检查。只有NPU真实验证才能证明性能收益。

第三轮预算：C21/C25/C26重新目标构建，36功能子检查成功后18批原10项50/200/5交错筛选，post-identity保持；所有失败保留。只有候选完整初筛通过才20新进程确认、full回归和正式交付更新。禁止用C21已通过的两批宣称默认分配器已稳定达标。


第三轮C21/C25/C26均36功能通过、18批原协议筛选完成，前后身份不变，无post-identity error；默认仍约82–84µs，1g约73µs，全部新候选未晋升。C25均匀槽分布没有闭合问题，C26字面入口SDK.text从9704增至17604字节、实际耗时无足够改善，不把“机器码变化”当成优化收益。480原件归档be9f4058b99dcdc0d180a77559915e3d125485470ad3ee0c3c49dc3de1f64f8b，逐SHA核验；见spaced_write_screen_results.json。

第四轮转向读写调度，缓存字节数仍严格与C25相同：
- C27 paired_history_pad：把对应K/V输出块各23552元素放在同一个47104-element UB槽内，MTE2用DataCopyPad Ext blockCount=2从两个已验证源区间成对读取（GM步长字节、UB步长32B）；QKV追加也按两组对应数据读取。只有目标大geometry、输出512B对齐、源K<V且差值在[47104,UINT32_MAX]并满足dtype对齐时启用，否则执行原C4/已定义Keychain路径。虚拟跨距内的间隙不访问、不分配；两输入仍由原封装recordStream保护。两半输出分别写入，W事件在两次store后发布，R事件在全部2-block读后发布；原双槽生命周期保持。CPU逐元素及显式2D地址/三种src-gap/UB两平面边界/正逆序和overrange guard验证通过。预计读取DMA指令1936（C25为2448）、写入2852（C25为1426），字节数/缓存footprint无变化，性能收益未测。
- C29 interleaved_history：按同一逻辑块交替预取/写回Key、Value，保留47104元素单平面DMA、不增加指令总数，也不使用跨分配区间的2D视图。每核14或15个完整K/V对（28/30任务），mask与缓存字节数同C25，CPU全67141632元素检查通过。
- C28 paired_history_aligned仅CPU准备，增加按已验证32B约束的基本DataCopy输出变体，暂不优先占用NPU。

已复核本地9.1.0 API页183–192/238–248：A3支持uint16，DataCopyPad Ext GM起址无对齐限制、Local起址32B、blockCount2合法、blockLen≤47104、GM srcStride为byte且uint32、UB dstStride为32B。基本DataCopy count*sizeof(uint16)全部为512B倍数，不发生向下取整。追加source-layout12（两dtype×正向2B源间隔/反序/相同源×ACLNN/direct），补强2D快路径和回退的真实验证；不增加官方case分母。预计C25/C27/C29目标构建→40子功能→18完整10项原协议/post-identity；通过才20新进程确认/full回归及正式报告/包更新。

第四轮40功能全部通过、18原协议性能完成，前后source/artifact一致，post-identity五库核验均成功。C27默认三批007=71.770/72.279/86.826µs、017=71.462/72.019/87.823µs，第三批失败；1g三批约60–61µs通过。C29默认约89µs失败，1g约61µs。confirmation gate按全部六批通过条件阻止后续confirm/full，没有选取前两批晋升；正式报告、XLSX、交付包和默认源码保持原样。

追加4进程memory-history诊断（NOT ACCEPTANCE），每进程31556次alloc匹配完整10项协议，每大case的1000次计时输出地址三元组唯一；四进程的007/017输入和输出绝对虚拟地址全部相同，K/V间隔均69206016B，均满足C27源地址guard。插桩耗时仍出现约71–73与87–88µs两档，故地址顺序回退不能解释差异；未直接采集设备分支轨迹，不把CPU guard推演称为设备分支观察。

另5个同进程history+PMU诊断：两个default L2进程的007 task均值72.014/86.713µs，aiv_r0_read_cache_miss_allocate均值45287/101184，r1相近；page1g约60.265µs、r0约379。两个default Pipe进程007 task约72.103/87.465µs，Scalar约5.415/5.366µs，MTE2约54.973/65.374µs、MTE3约52.281/60.730µs。各数字仅属于各自进程，不将L2和Pipe拼接成一个样本，也不从计数推断未测物理地址/cache hash。791原件归档13faae5e779bea2574f093a43618ca3eff1ac28fa6a08e1591c4976bd9f028fd，含原始CSV和memory trace，逐文件SHA核验；采样均不参与验收。见history_scheduling_screen_results.json。

第五轮C30仅将C27配对块由normal/bypass分组改为线性物理输出地址顺序，各块仍按C25原mask选择Key hint；验证过早写回是否使尚未读取的输入遭到额外置换这一假设。1936读指令、2852写指令、正常Key写54706176B/旁路写79577088B、184KiB UB、POD和事件生命周期全部相同；实际CPU mapper全67141632元素/1426块、29–30任务每核检查通过。C28基本DataCopy整块输出作为独立对照，满足既有512B字节倍数约束；不把新增API变体当成已优化。计划C27/C28/C30构建→42功能→18完整原协议/post-identity筛选；任一批失败不晋升，通过才20新进程确认和full回归。

备用C31仅CPU准备：保持C30线性成对读取、双槽和事件生命周期，将Key写准入改为隔一个父tile正常写一次，正常Key写33603584B、旁路总写100679680B；两份历史读取仍全normal。正常驻留从约180MiB降至约160MiB，用增加20MiB旁路写来检验能否压低已观测到的额外输入读miss，不把miss计数直接折算成HBM字节。与此前只减少约4MiB的C22不同，保留成对读取和线性调度，仅测试一个较大余量边界；不安排无限准入参数扫描。CPU实际helper全67141632元素、1426块、1936读/2852写指令检查通过，28候选通用mapper和9项driver测试通过。尚无目标构建或NPU结果；只有第五轮仍失败时再决定是否实际投入该单独候选。新runner会按metadata中的真实周期生成边界说明，不覆写当前运行的冻结runner。

第五轮42功能子检查全部通过、18筛选完成；C30三个default筛选批均通过，C27/C28各有两批default超限。按原计划对C30执行20个新进程确认，结果default 5/10通过、5/10失败：007交替约72与102–103µs，017约71–72与102–104µs；page1g 10/10通过、约60µs。两组全部10个性能case均完整，所有post-identity检查成功、前后源码/产物一致。C30不晋升，full回归按门槛未启动，不能用三批筛选或五个快进程更新正式交付。639原件归档ef7b288e5d11d58b860f5d4335d638611019408dc006b19059ebf115c3beeab5，逐文件SHA核验，见linear_paired_screen_results.json。

第六轮已按条件门槛启动C30/C31独立重建和28功能检查，成功后12筛选；只有候选全部六批通过才20确认/full。根目录/workspace/t5qkv-half-admission-20261114-01，bundle5e9fc7ad272571ba80a0a316ef6780c6e6ba5fa4b9b72de51aa6b1315b3e3073，C31此时只是完成目标构建，尚无性能通过结论。

备用C32改为读写缓存互补，父版本为C29：Value历史读取全normal、Value输出全bypass；对每个逻辑Key输出块，Key输出normal时历史Key读取bypass，Key输出bypass时历史Key读取normal。避免相同逻辑区间的Key输入和独立输出同时占用L2，将正常驻留的相对512B-line上界降至134425088B（约128.2MiB），同时将名义旁路流量分成约52MiB读和76MiB写，检验能否避免单侧MTE3写流量过大以及额外输入miss。该上界不是物理cache映射或实测HBM流量证明。与C9全部输出bypass不同，此方案有正常Key写并把部分只读Key输入主动旁路；与C19四平面部分准入不同，此处每个Key区间的输入/输出hint互补，保留Value输入全正常。

C32沿用C29单平面双槽、交替K/V完整块调度及2448读取piece/1426写块、184KiB UB、52字节POD、事件生命周期和输出分配；无2D跨分配源视图或源顺序限制。只读输入的normal/bypass视图不写入输入，Key输出hint分界仍512B整块；其他几何/未对齐输出预选原支持路径。CPU实际helper全67141632元素检查、正常/旁路读写字节和相对cache-line覆盖上界检查通过，29候选通用CPU及9项driver测试通过。此时仅CPU准备，没有NPU构建/通过结论；只有第六轮性能仍失败时投入C29/C32两版本、28功能及12原协议筛选。原CPU JSON记录的C25读取piece一直为2448，先前说明中的2446笔误已修正，原件未修改。

第六轮28功能全部通过、12筛选完整结束。C31默认三批007=79.611/79.324/79.816µs、017=80.363/80.006/79.742µs，均失败；page1g三批约74µs通过。控制器误把此前已淘汰的对照C30列为可确认候选，额外执行20个进程；原件保留，default仍5/10失败，page1g另有001=30.098µs超过28µs，不能覆盖第五轮C30的失败。所有source/artifact/post-identity检查成功。525原件归档84515aefb806535f655ca3ab6ac2d5c199891ab9f564340f0a4e22691616b782逐SHA核验，见half_admission_screen_results.json。

第七轮原等待控制器在任何build/NPU操作前终止并保留；独立driver02仅允许新C32确认，新的gate/plan/log文件避免覆写原件。C29/C32目标构建已完成，正在28功能，随后12筛选；C32全部六批通过才20确认/full。根/workspace/t5qkv-complementary-cache-20261114-01，bundle8f3dfc261d575cae16b9a4242948a9f5bfe825caa31051da25a0cb94f63ec7aa。

备用C33仅CPU准备：保持C32全部hint/数据量/184KiB UB/事件，将16个Query块从历史任务较多的前16核，分配到只处理14对而非15对K/V块的第41–47核，每核2或3个Query块。仍使用原decode_task/copy_segment，其他路径原Query分配。CPU全67M历史元素与16个Query任务/地址/覆盖检查、30候选通用CPU及9项driver检查通过。只有C32剩余实测差距≤3µs时，才考虑该负载均衡变体的目标测试；它不解决十几微秒的缓存退化，当前没有NPU通过结论。

第七轮28功能全部通过、12原协议筛选完成，C32六批全部失败，default大case约106.6–107.0µs、page1g约105.5–105.8µs，另一次001=31.362µs超28µs。实际加载五库/source/artifact检查成功。C33条件门槛因此未启动任何build/NPU。四个同进程history+PMU诊断完整：default007 L2 task106.632µs/r0 miss-allocate72，page1g105.310µs/r0为0；另两个Pipe进程default task106.560µs、Scalar5.354、MTE286.507、MTE368.277，page1g105.315/5.349/86.113/67.396。低miss不等于没有HBM读取：该计划主动旁路约52MiB Key读取。L2/Pipe各属于自己的进程，不拼接成同一次样本。该计划虽消除主要分配策略差距，计划旁路读的代价未达绝对耗时要求，不晋升。见complementary_cache_screen_results.json及其逐SHA证据归档。

C34以此作一次结构验证，并不声称512B对齐必然提升带宽：改用60KiB逻辑块、三个独立UB槽，每槽前后各512B私有空间，总183KiB。Key输入基址512B对齐时，将历史Key旁路读向512B边界扩展，直接落入槽前缀/后缀；只有首个historical piece可能带前缀，后续head piece前缀为0，后续QKV/head覆盖先前后缀，最终store只写真实有效块。输入总长度512B倍数、向前/向后扩展均在合法只读输入区间内；未对齐输入不扩读。完整CPU local-map模拟覆盖67M元素及所有额外读取边界，额外读取341632B、无额外GM。每核44/46次历史任务，16Query分至37–47短核；52字节POD不变，其余几何/未对齐输出沿用原双槽路径。

三槽各自申请MTE2_MTE3/MTE3_MTE2事件，先预取最多三块，再依次等R/store/setW，只有等对应槽W后才重填；最终全部W消费后才复用UB处理Query并释放事件。A3事件ID范围0–7已核对官方9.1页1120，三份/方向合法，TPipe只有一个TBuf。CPU槽复用依赖/地址/Query覆盖及31候选通用mapper、9项driver测试通过。C34仍待目标构建；先完整20个功能子检查（含queue/tail/hot），再6批原协议筛选，全部通过才20新进程确认。已在前置full中覆盖的相同源码/产物不重复执行full。若失败停止该复合方向，不通过扩大参数扫描或改变分配器来宣布通过。
