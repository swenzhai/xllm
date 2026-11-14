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
  T5SplitQkvAppendCacheTilingData tiling{};
  assert(t5qkv::make_tiling(/*batch=*/16,
                            /*head_num=*/16,
                            /*past_length=*/2048,
                            /*head_dim=*/64,
                            /*available_vector_cores=*/48,
                            tiling));
  assert(!t5qkv::make_read_balance_policy(tiling).enabled);
  assert(tiling.cacheable_key_rows == 240 && t5qkv::fuses_append(tiling));
  std::array<uint32_t, 768> seen{};
  uint64_t normal_elements = 0;
  uint64_t bypass_elements = 0;
  for (uint32_t core = 0; core < 48; ++core) {
    for (uint32_t phase = 0; phase < 2; ++phase) {
      const auto range = t5qkv::partition_tasks(
          phase == 0 ? 0 : 720, phase == 0 ? 720 : 48, 48, core);
      const uint32_t physical =
          t5qkv::cache_write_phase_task_start(tiling, phase, range.begin);
      for (uint32_t rank = range.begin; rank < range.end; ++rank) {
        const uint32_t actual = physical + rank - range.begin;
        assert(actual < seen.size());
        ++seen[actual];
        const uint32_t row = actual / 3;
        const bool expected_normal = (row >= 4 && row < 84) ||
                                     (row >= 88 && row < 168) ||
                                     (row >= 172 && row < 252);
        assert(expected_normal == (phase == 0));
        const uint32_t tile = actual % 3;
        const uint32_t remaining = 2048 * 64 - tile * tiling.tile_elements;
        const uint32_t elements =
            (remaining < tiling.tile_elements ? remaining
                                              : tiling.tile_elements) +
            (tile == 2 ? 64 : 0);
        normal_elements += phase == 0 ? elements : 0;
        bypass_elements += phase == 1 ? elements : 0;
      }
    }
  }
  for (const uint32_t count : seen) {
    assert(count == 1);
  }
  assert(normal_elements == uint64_t{240} * 2049 * 64);
  assert(bypass_elements == uint64_t{16} * 2049 * 64);
  for (const uint32_t row : {0, 4, 84, 88, 168, 172, 252, 256}) {
    assert((uint64_t{row} * 2049 * 64 * 2) % 512 == 0);
  }
  std::array<uint32_t, 13> words{};
  std::memcpy(words.data(), &tiling, sizeof(tiling));
  std::cout << "{\"scope\":\"host_write_mapping_only\",\"npu_executed\":false,"
               "\"tasks\":768,\"class_boundaries_aligned_bytes\":512,"
               "\"normal_key_write_bytes\":62945280,"
               "\"bypass_key_write_bytes\":4196352,\"tiling_words\":[";
  for (uint32_t i = 0; i < words.size(); ++i) {
    std::cout << (i == 0 ? "" : ",") << words[i];
  }
  std::cout << "],\"status\":\"passed\"}\n";
}
