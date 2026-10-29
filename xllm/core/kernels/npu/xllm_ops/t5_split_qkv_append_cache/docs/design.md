<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# T5 QKV split and cache append design

## Contract

Inputs are dense FP16 or BF16 `qkv[B,1,3HD]`, `pastKey[B,H,L,D]`, and
`pastValue[B,H,L,D]`. The packed dimension contains all Q, then all K, then all V.
The integer attributes `headNum` and `headDim` must match H and D; their defaults
are 1. Bounds are B=1..64, H=1..128, D=1..256, L=0..8192, and
`B*H*D*(2*L+3) <= 2^28` total output elements.

Every call allocates three separate contiguous tensors: query `[B,H,1,D]`, key
and value `[B,H,L+1,D]`. The implementation performs only uint16 data movement;
FP16/BF16 NaNs, infinities, signed zero, and all other bit patterns are preserved.
It does not modify inputs, read an empty history, alias storage, convert floating
point values, or cache outputs. Continuous offset views are supported.

The ACL tensor descriptors always use ND. The framework also accepts torch_npu's
NCHW label for ordinary contiguous rank-four storage, which has the same linear
layout here. Packed device formats are rejected.

## Address mapping and scheduling

For head row `r=b*H+h`, historical input starts at `r*L*D`, and its output starts
at `r*(L+1)*D`. The appended token starts at `r*(L+1)*D+L*D`. Source QKV offsets
are `b*3HD + hD`, `b*3HD + HD+hD`, and `b*3HD + 2HD+hD`.

The implementation slices each historical K/V head into 47104-element tiles.
When the final history tile has a 32-byte-aligned UB tail and enough space for
its padded append token, the token is loaded into that tail and written back in
one DMA with the history. The host `can_fuse_append` guard checks both conditions;
otherwise append tokens retain independent tasks. The device reads the selected
task layout from the total task count, avoiding repeated tail divisions. L=0 creates no history tasks.
Fused tasks keep history and packed-QKV reads separate, with adjacent output
ranges checked by the host mapping test.
Q tasks copy contiguous HD-sized batch segments. Historical K is split at the
cache-admission row boundary into normal-cache and bypass groups; V is a third
group. Each group, fallback append and Q distribute contiguous task ranges across
all available Vector Cores, capped at 48. This balances HBM traffic without the
large per-task page strides of a cyclic assignment. Each output element has exactly one
writer.

`common/t5_split_qkv_append_cache_tiling.h` defines the 44-byte tiling POD and the
same task decoder used by the device and the host mapping checker. The host
uses checked dimension bounds before constructing products and rejects excessive
output sizes. Both ACLNN and direct launch use the same kernel class.

## Memory and synchronization

Two 92 KiB UB buffers are bound to a VECIN/VECOUT queue. A copy allocates a buffer,
performs GM-to-UB DataCopyPad, enqueues/dequeues it, performs UB-to-GM DataCopyPad,
and releases it. History copying prefills one task, then fills the next queue
slot before retiring the current task, overlapping MTE2 and MTE3. Total UB is
184 KiB, leaving 8 KiB below the reported 192 KiB capacity. Queue dependencies protect buffer reuse. Every transfer begins
at a 32-byte-aligned UB address. A fused append uses a checked aligned tail;
the fallback append uses the buffer base. GM transfers have exact byte lengths and do
not write UB padding to neighbouring rows. Scalar GM stores are not used.

Explicit global workspace is zero. Output bytes are
`2*B*H*D*(2*L+3)`. The complete GM read/write volume is twice this quantity; L2 hits reduce HBM traffic.
The kernel uses no intermediate device tensor or conversion buffer. ACLNN tiling
uses runtime launch metadata; the direct kernel receives the small POD by value.

For large working sets, the ACLNN kernel uses a measured 172 MiB L2 budget,
leaving headroom below the target platform's reported 192 MiB capacity. This is
a cache-policy tuning parameter, not a new allocation or workspace. With
`N=B*H*D`, the complete input/output working set is `8*N*L+12*N` bytes. Sets
within the budget use normal caching. For larger sets, V output uses the disable
hint; the kernel caches a prefix of whole K rows using the budget remaining
after `4*N*(L+2)` bytes for the historical inputs, packed QKV and Q output.
The rest of K output uses the disable hint. `make_cache_policy` returns both
the admitted element prefix and its history-task boundary; the kernel and host
mapping checker share this policy and the five-phase partition.

Two K GlobalTensor aliases select the appropriate cache hint without changing
the GM address, copy length or ownership. Both aliases are initialized once;
each task selects one by comparing its first output offset with the cached
prefix length. No per-task cache-hint setup or row division is needed. The
legacy direct-launch build does not enable CANN's L2-hint runtime configuration;
it uses the same task mapping and bit-copy kernel with normal caching. The
official performance protocol measures the ACLNN `run` entry.

The Python extension obtains the current stream on the input device without
draining the framework host queue. It submits both paths through torch_npu's
`OpCommand::RunOpApi`, preserving the order of preceding framework operations.
It records input storages on that stream before enqueuing the launch. The
torch_npu caching allocator puts completion events for subsequent input frees
behind the already queued launch, including when the caller releases an input
on another stream. Fresh outputs belong to the current allocation stream and
can only be reused on that same stream, whose subsequent work follows this
launch. They need no redundant origin-stream events. Fixed-size metadata
describes each view without retaining or destroying Python-associated tensor
or storage owners on the submission worker. This follows torch_npu 2.10.0's
native caching allocator stream semantics; external allocation/IPC lifetimes
remain the external owner's responsibility.
The two-stage ACLNN query must return zero workspace; the normal non-reusable
executor is consumed by the execution stage. Tensor descriptors live through
both stages. Worker failures use C++ exceptions without accessing Python.

## Interfaces

- `run(inputs, attributes)` validates metadata, allocates fresh outputs, then calls
  `aclnnT5SplitQkvAppendCacheGetWorkspaceSize` and `aclnnT5SplitQkvAppendCache`.
- `run_direct(inputs, attributes)` validates and allocates identically, then
  invokes `t5qkv_launch` from the separately built direct kernel library.
- `prepare(inputs, attributes)` returns a callable accepting new inputs with
  identical metadata. It retains no input tensor, output, device pointer or
  stream. Every prepared call still validates and allocates fresh outputs.
- `validate_out` exposes output metadata/storage validation for engineering tests;
  `workspace_size` exposes the static zero-workspace contract, which is also
  checked against every real ACLNN workspace query.
- The xLLM C++ wrapper uses the existing `EXEC_NPU_CMD` integration and is declared
  in `xllm_ops_api.h`. Building the standalone project does not build the complete
  xLLM inference server.

## Validation boundaries

The public verifier covers 35 inputs and the official benchmark covers ten of
those with the unchanged 50-warmup, 200-iteration, five-round protocol. Extended
tests separately cover direct calls, extra seeds and offsets, illegal metadata,
stream ordering, early release, output ownership, repeat append and memory.
The host mapping checker proves interval/source mapping properties only; it
cannot establish device correctness or performance.

The supplied task book has a checksum mismatch and conflicting relative versus
absolute performance language. Development follows the executable absolute-time
policy without altering the package. The available machine matches SoC, CANN,
Python and framework versions, but its Ubuntu distribution and driver differ
from the acceptance environment. Results on it are development evidence.


## Same-device comparison

`tests/taskbook_baseline.py` implements the taskbook's sole candidate,
`views_concat_q_copy`. `tests/compare_same_device.py` first qualifies that
candidate with all 35 public accuracy cases. It then runs the unchanged public
benchmark separately for the custom implementation and reference on device 0,
using identical fixture values and the original 50/200/5 protocol. Independent
processes prevent one algorithm's temporary allocations from setting up the
other's allocator state. Device addresses are not required to be identical.
Reference absolute-threshold failures are retained as original child exit codes;
those submission limits do not qualify reference performance. Current/reference
ratios and all five wall/event samples are retained. Earlier mixed-process
experiments, including failures, remain diagnostic evidence.
