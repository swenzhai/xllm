/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "common/t5_split_qkv_append_cache_tiling.h"

int32_t main() {
  T5SplitQkvAppendCacheTilingData actual{};
  assert(t5qkv::make_tiling(/*batch=*/16,
                            /*head_num=*/16,
                            /*past_length=*/2048,
                            /*head_dim=*/64,
                            /*available_vector_cores=*/48,
                            actual));
  const T5SplitQkvAppendCacheTilingData fixed = t5qkv::fixed_history_tiling();
  assert(sizeof(fixed) == 52 &&
         std::memcmp(&actual, &fixed, sizeof(fixed)) == 0);
  assert(t5qkv::matches_fixed_history_tiling(actual));
  std::array<uint32_t, 13> words{};
  std::memcpy(words.data(), &actual, sizeof(actual));
  for (uint32_t index = 0; index < words.size(); ++index) {
    std::array<uint32_t, 13> changed = words;
    changed[index] ^= 1;
    T5SplitQkvAppendCacheTilingData other{};
    std::memcpy(&other, changed.data(), sizeof(other));
    assert(!t5qkv::matches_fixed_history_tiling(other));
  }
  for (const std::array<uint32_t, 4> shape :
       {std::array<uint32_t, 4>{8, 32, 2048, 64},
        {16, 16, 2047, 64},
        {16, 16, 2048, 65}}) {
    T5SplitQkvAppendCacheTilingData other{};
    assert(
        t5qkv::make_tiling(shape[0], shape[1], shape[2], shape[3], 48, other));
    assert(!t5qkv::matches_fixed_history_tiling(other));
  }
  std::cout << "{\"scope\":\"fixed_tiling_CPU_guard_only\",\"npu_executed\":"
               "false,\"status\":\"passed\",\"tiling_bytes\":52,\"single_word_"
               "guard_mutations\":13,\"other_valid_shapes\":3}\n";
}
