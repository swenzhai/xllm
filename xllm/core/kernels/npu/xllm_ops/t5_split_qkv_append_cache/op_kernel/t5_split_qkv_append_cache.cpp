/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "op_kernel/t5_split_qkv_append_cache_kernel.h"

#if defined(T5QKV_DIRECT_LAUNCH)
// The direct path passes the small POD by value; no device tiling allocation or
// global workspace is required. Both paths instantiate the exact same kernel.
extern "C" __global__ __aicore__ void t5_split_qkv_append_cache_direct_kernel(
    GM_ADDR qkv,
    GM_ADDR past_key,
    GM_ADDR past_value,
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    T5SplitQkvAppendCacheTilingData tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  t5qkv::KernelT5SplitQkvAppendCache op;
  op.init(qkv, past_key, past_value, query, key, value, tiling);
  op.process();
}

#else
extern "C" __global__ __aicore__ void t5_split_qkv_append_cache(
    GM_ADDR qkv,
    GM_ADDR past_key,
    GM_ADDR past_value,
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR /*workspace*/,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(T5SplitQkvAppendCacheTilingData);
  GET_TILING_DATA_WITH_STRUCT(
      T5SplitQkvAppendCacheTilingData, tiling_data, tiling);
  t5qkv::KernelT5SplitQkvAppendCache op;
  op.init(qkv, past_key, past_value, query, key, value, tiling_data);
  op.process();
}
#endif
