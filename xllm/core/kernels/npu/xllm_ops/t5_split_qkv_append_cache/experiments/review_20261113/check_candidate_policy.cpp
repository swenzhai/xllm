/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
==============================================================================*/

#include <array>
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
  assert(tiling.tile_elements == 47104 && tiling.block_dim == 48);
  assert(tiling.cache_tiles_per_head == 3 && t5qkv::fuses_append(tiling));
  const auto cache_policy = t5qkv::make_cache_policy(tiling);
  const uint32_t value_rows = policy.enabled ? policy.value_rows : 256;
  const uint32_t key_rows = policy.enabled ? 256 : tiling.cacheable_key_rows;
  const uint64_t budget =
      policy.enabled ? t5qkv::kReadBalanceBudgetBytes : t5qkv::kL2BudgetBytes;
  assert(value_rows % 16 == 0 && key_rows % 16 == 0);
  const uint64_t normal_bytes =
      2 * uint64_t{16 * 16 * 64} * 2048 + 2 * uint64_t{value_rows} * 2048 * 64 +
      2 * uint64_t{key_rows} * 2049 * 64 + 8 * uint64_t{16 * 16 * 64};
  assert(normal_bytes <= budget);
  // All three history classes have equal per-core task counts at this geometry.
  const std::array<uint32_t, 3> phase_counts =
      policy.enabled ? std::array<uint32_t, 3>{tiling.cache_tasks_per_tensor,
                                               policy.value_tasks,
                                               tiling.cache_tasks_per_tensor -
                                                   policy.value_tasks}
                     : std::array<uint32_t, 3>{cache_policy.key_tasks,
                                               tiling.cache_tasks_per_tensor -
                                                   cache_policy.key_tasks,
                                               tiling.cache_tasks_per_tensor};
  for (const uint32_t count : phase_counts) {
    assert(count % tiling.block_dim == 0);
  }
  if (!policy.enabled && t5qkv::kL2BudgetBytes == uint64_t{172} * 1024 * 1024) {
    assert(tiling.cacheable_key_rows == 0 && cache_policy.key_tasks == 0);
  } else if (!policy.enabled) {
    T5SplitQkvAppendCacheTilingData boundary{};
    assert(t5qkv::make_tiling(/*batch=*/4,
                              /*head_num=*/128,
                              /*past_length=*/1023,
                              /*head_dim=*/65,
                              /*available_vector_cores=*/48,
                              boundary));
    assert(boundary.cacheable_key_rows > 0 &&
           boundary.cacheable_key_rows < 512);
    assert(boundary.cacheable_key_rows ==
           (t5qkv::kL2BudgetBytes == uint64_t{186} * 1024 * 1024 ? 432 : 456));
    const uint64_t boundary_bytes = uint64_t{boundary.cacheable_key_rows} *
                                    (boundary.past_length + 1) *
                                    boundary.head_dim * sizeof(uint16_t);
    assert(boundary_bytes % 512 == 0);
    // At length 1024 the row/alignment/core grouping admits no complete group.
    // This must become an all-bypass Key-output plan without invalid ranges.
    assert(t5qkv::make_tiling(/*batch=*/4,
                              /*head_num=*/128,
                              /*past_length=*/1024,
                              /*head_dim=*/65,
                              /*available_vector_cores=*/48,
                              boundary));
    assert(boundary.cacheable_key_rows == 0);
  }
  std::cout << "{\"npu_executed\":false,\"scope\":\"host_policy_only\","
            << "\"tiling_bytes\":" << sizeof(tiling)
            << ",\"budget_bytes\":" << budget
            << ",\"value_rows\":" << value_rows
            << ",\"normal_key_rows\":" << key_rows
            << ",\"read_balance_enabled\":"
            << (policy.enabled ? "true" : "false")
            << ",\"value_tasks\":" << value_rows * tiling.cache_tiles_per_head
            << ",\"normal_bytes\":" << normal_bytes << "}\n";
  return 0;
}
