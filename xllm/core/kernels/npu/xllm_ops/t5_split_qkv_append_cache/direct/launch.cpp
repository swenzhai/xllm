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

#include "common/t5_split_qkv_append_cache_tiling.h"

// The CANN-generated launcher copies this host POD into the kernel argument
// packet synchronously; the pointer is not a device workspace allocation.
extern "C" uint32_t aclrtlaunch_t5_split_qkv_append_cache_direct_kernel(
    uint32_t block_dim,
    void* stream,
    void* qkv,
    void* past_key,
    void* past_value,
    void* query,
    void* key,
    void* value,
    T5SplitQkvAppendCacheTilingData* tiling);

extern "C" uint32_t t5qkv_launch(uint32_t block_dim,
                                 void* stream,
                                 void* qkv,
                                 void* past_key,
                                 void* past_value,
                                 void* query,
                                 void* key,
                                 void* value,
                                 T5SplitQkvAppendCacheTilingData tiling) {
  return aclrtlaunch_t5_split_qkv_append_cache_direct_kernel(
      block_dim, stream, qkv, past_key, past_value, query, key, value, &tiling);
}
