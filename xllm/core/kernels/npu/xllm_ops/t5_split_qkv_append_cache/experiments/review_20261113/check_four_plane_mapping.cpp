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
  const uint32_t rows = t5qkv::four_plane_rows(tiling);
  assert(rows == 184 || rows == 176);
  assert((rows * 2049 * 64 * 2) % 512 == 0);
  std::array<std::array<uint32_t, 768>, 2> seen{};
  uint64_t normal_read = 0;
  uint64_t normal_write = 0;
  uint64_t bypass_read = 0;
  uint64_t bypass_write = 0;
  uint32_t appends = 0;
  for (uint32_t core = 0; core < 48; ++core) {
    uint32_t core_tasks = 0;
    for (uint32_t phase = 0; phase < 4; ++phase) {
      const auto range = t5qkv::four_plane_range(tiling, phase, core);
      core_tasks += range.end - range.begin;
      for (uint32_t rank = range.begin; rank < range.end; ++rank) {
        assert(rank < 768);
        ++seen[phase / 2][rank];
        const uint32_t row = rank / 3;
        assert((row < rows) == ((phase & 1) == 0));
        const auto task = t5qkv::decode_task(tiling, phase / 2 * 768 + rank);
        assert(task.source_offset ==
               row * 2048 * 64 + rank % 3 * tiling.tile_elements);
        assert(task.destination_offset ==
               row * 2049 * 64 + rank % 3 * tiling.tile_elements);
        assert(task.count + task.append_count <= tiling.tile_elements);
        assert(task.source_offset + task.count <= 256 * 2048 * 64);
        assert(task.destination_offset + task.count + task.append_count <=
               256 * 2049 * 64);
        if ((phase & 1) == 0) {
          normal_read += task.count * 2;
          normal_write += (task.count + task.append_count) * 2;
        } else {
          bypass_read += task.count * 2;
          bypass_write += (task.count + task.append_count) * 2;
        }
        appends += task.append_count != 0;
      }
    }
    assert(core_tasks == 32);
  }
  for (const auto& tensor : seen) {
    for (const uint32_t count : tensor) {
      assert(count == 1);
    }
  }
  assert(appends == 512);
  assert(normal_read == uint64_t{2} * rows * 2048 * 64 * 2);
  assert(normal_write == uint64_t{2} * rows * 2049 * 64 * 2);
  assert(normal_read + bypass_read == uint64_t{2} * 256 * 2048 * 64 * 2);
  assert(normal_write + bypass_write == uint64_t{2} * 256 * 2049 * 64 * 2);
  assert(normal_read + normal_write + 131072 < uint64_t{190} * 1024 * 1024);
  std::array<uint32_t, 13> words{};
  std::memcpy(words.data(), &tiling, sizeof(tiling));
  tiling.head_dim = 65;
  assert(t5qkv::four_plane_rows(tiling) == 0);
  std::cout << "{\"scope\":\"four_plane_CPU_mapping_only\",\"npu_executed\":"
               "false,\"status\":\"passed\",\"rows\":"
            << rows
            << ",\"tasks\":1536,\"tasks_per_core\":32,\"appends\":" << appends
            << ",\"normal_read_bytes\":" << normal_read
            << ",\"normal_write_bytes\":" << normal_write
            << ",\"bypass_read_bytes\":" << bypass_read
            << ",\"bypass_write_bytes\":" << bypass_write
            << ",\"tiling_words\":[";
  for (uint32_t i = 0; i < words.size(); ++i) {
    std::cout << (i ? "," : "") << words[i];
  }
  std::cout << "]}\n";
}
