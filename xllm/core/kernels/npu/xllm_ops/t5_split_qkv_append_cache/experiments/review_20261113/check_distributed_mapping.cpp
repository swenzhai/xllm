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
  assert(policy.enabled && policy.value_tasks == 720);
  std::array<uint32_t, 768> seen{};
  uint64_t normal_elements = 0;
  uint64_t bypass_elements = 0;
  uint32_t appends = 0;
  for (uint32_t core = 0; core < 48; ++core) {
    for (uint32_t phase = 1; phase < 3; ++phase) {
      const uint32_t begin = phase == 1 ? 0 : policy.value_tasks;
      const uint32_t count = phase == 1 ? policy.value_tasks : 48;
      const auto range =
          t5qkv::partition_tasks(begin, count, /*cores=*/48, core);
      const uint32_t physical = t5qkv::read_balance_phase_task_start(
          tiling, policy.value_tasks, phase, range.begin);
      assert(physical == 16 * core + (phase == 1 ? 0 : 15));
      assert(range.end - range.begin == (phase == 1 ? 15 : 1));
      for (uint32_t task = range.begin; task < range.end; ++task) {
        const uint32_t actual = physical + task - range.begin;
        assert(actual < seen.size());
        ++seen[actual];
        const uint32_t tile = actual % 3;
        const uint32_t remaining = 2048 * 64 - tile * tiling.tile_elements;
        const uint32_t elements =
            remaining < tiling.tile_elements ? remaining : tiling.tile_elements;
        normal_elements += phase == 1 ? elements : 0;
        bypass_elements += phase == 2 ? elements : 0;
        appends += tile == 2 ? 1 : 0;
      }
    }
  }
  for (const uint32_t count : seen) {
    assert(count == 1);
  }
  assert(normal_elements == uint64_t{240} * 2048 * 64);
  assert(bypass_elements == uint64_t{16} * 2048 * 64);
  assert(appends == 256);
  assert(t5qkv::read_balance_phase_task_start(tiling, 720, 0, 42) == 42);
  assert(t5qkv::read_balance_phase_task_start(tiling, 719, 1, 42) == 42);
  tiling.block_dim = 24;
  assert(t5qkv::read_balance_phase_task_start(tiling, 720, 1, 42) == 42);
  std::cout << "{\"scope\":\"host_distributed_cursor_coverage_only\","
               "\"npu_executed\":false,\"tasks\":768,\"unique_appends\":256,"
               "\"normal_bytes\":62914560,\"bypass_bytes\":4194304,"
               "\"status\":\"passed\"}\n";
  return 0;
}
