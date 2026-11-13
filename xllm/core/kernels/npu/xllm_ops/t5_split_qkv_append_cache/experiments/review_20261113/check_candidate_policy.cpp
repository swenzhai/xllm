/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/

#include <cassert>
#include <cstdint>
#include <iostream>

#include "common/t5_split_qkv_append_cache_tiling.h"

int32_t main() {
  T5SplitQkvAppendCacheTilingData tiling{};
  assert(t5qkv::make_tiling(/*batch=*/16,
                            /*head_num=*/16,
                            /*past_length=*/2048,
                            /*head_dim=*/64,
                            /*available_vector_cores=*/48,
                            tiling));
  const auto policy = t5qkv::make_read_balance_policy(tiling);
  assert(policy.enabled);
  assert(tiling.tile_elements == 47104 && tiling.block_dim == 48);
  assert(tiling.cache_tiles_per_head == 3 && t5qkv::fuses_append(tiling));
  assert(policy.value_rows % 16 == 0);
  const uint64_t normal_bytes =
      2 * uint64_t{16 * 16 * 64} * (2 * uint64_t{2048} + 5) +
      2 * uint64_t{policy.value_rows} * 2048 * 64;
  assert(normal_bytes <= t5qkv::kReadBalanceBudgetBytes);
  // All three history classes have equal per-core task counts at this geometry.
  assert(tiling.cache_tasks_per_tensor % tiling.block_dim == 0);
  assert(policy.value_tasks % tiling.block_dim == 0);
  assert((tiling.cache_tasks_per_tensor - policy.value_tasks) %
             tiling.block_dim ==
         0);
  std::cout << "{\"npu_executed\":false,\"scope\":\"host_policy_only\","
            << "\"tiling_bytes\":" << sizeof(tiling)
            << ",\"budget_bytes\":" << t5qkv::kReadBalanceBudgetBytes
            << ",\"value_rows\":" << policy.value_rows
            << ",\"value_tasks\":" << policy.value_tasks
            << ",\"normal_bytes\":" << normal_bytes << "}\n";
  return 0;
}
