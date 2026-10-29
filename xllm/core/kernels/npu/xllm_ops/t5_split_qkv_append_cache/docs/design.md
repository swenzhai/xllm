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

The initial implementation slices each historical K/V head into 16384-element
tiles. The last tile appends the current token through a separate copy operation.
L=0 creates one append-only task per cache row. Q tasks copy contiguous HD-sized
batch segments, splitting only when required by the tile size. Tasks use cyclic
core assignment, bounded by the available Vector Core count and 48 cores on the
target SoC. Each output element has exactly one writer.

`common/t5_split_qkv_append_cache_tiling.h` defines the 44-byte tiling POD and the
same task decoder used by the device and the host mapping checker. The host
uses checked dimension bounds before constructing products and rejects excessive
output sizes. Both ACLNN and direct launch use the same kernel class.

## Memory and synchronization

Two 32 KiB UB buffers are bound to a VECIN/VECOUT queue. A copy allocates a buffer,
performs GM-to-UB DataCopyPad, enqueues/dequeues it, performs UB-to-GM DataCopyPad,
and releases it. Queue dependencies protect buffer reuse. Every segment begins
at a 32-byte-aligned UB base, including the append token; no append uses an
unaligned history-tail UB offset. GM transfers have exact byte lengths and do
not write UB padding to neighbouring rows. Scalar GM stores are not used.

Explicit global workspace is zero. Output bytes are
`2*B*H*D*(2*L+3)`. The ideal total GM read/write volume is twice this quantity.
The kernel uses no intermediate device tensor or conversion buffer. ACLNN tiling
uses runtime launch metadata; the direct kernel receives the small POD by value.

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
