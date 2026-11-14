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
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "common/t5_split_qkv_append_cache_tiling.h"

int32_t main(int32_t argc, char** argv) {
  const bool spaced = argc == 2 && std::strcmp(argv[1], "--evenly-spaced") == 0;
  assert(argc == 1 || spaced);
  T5SplitQkvAppendCacheTilingData tiling{};
  assert(t5qkv::make_tiling(/*batch=*/16,
                            /*head_num=*/16,
                            /*past_length=*/2048,
                            /*head_dim=*/64,
                            /*available_vector_cores=*/48,
                            tiling));
  assert(t5qkv::uses_flat_history(tiling));
  const uint32_t output_elements = 256 * 2049 * 64;
  const uint32_t tasks = t5qkv::dispersed_output_count(tiling);
  assert(tasks == 713);
  std::vector<uint8_t> key_seen(output_elements, 0);
  std::vector<uint8_t> value_seen(output_elements, 0);
  std::array<std::array<uint32_t, 713>, 2> seen{};
  uint64_t read_bytes = 0;
  uint64_t normal_write = 0;
  uint64_t bypass_write = 0;
  uint32_t pieces = 0;
  for (uint32_t core = 0; core < 48; ++core) {
    uint32_t core_tasks = 0;
    for (uint32_t phase = 0; phase < 3; ++phase) {
      const auto range = t5qkv::dispersed_output_range(tiling, phase, core);
      core_tasks += range.end - range.begin;
      uint32_t physical = t5qkv::dispersed_output_task(phase, range.begin);
      for (uint32_t rank = range.begin; rank < range.end; ++rank) {
        assert(physical == t5qkv::dispersed_output_task(phase, rank));
        assert(physical < tasks);
        ++seen[phase == 2 ? 1 : 0][physical];
        const uint32_t lane = physical % 16;
        const bool normal = spaced ? (physical >= tasks / 16 * 16 ||
                                      (lane != 4 && lane != 9 && lane != 14))
                                   : lane < t5qkv::kDispersedNormalSlots;
        assert(phase == 2 || normal == (phase == 0));
        const uint32_t offset = physical * tiling.tile_elements;
        const uint32_t count =
            std::min(output_elements - offset, tiling.tile_elements);
        assert(offset % 256 == 0 && count % 256 == 0 && count > 0);
        normal_write += phase == 0 ? count * 2 : 0;
        bypass_write += phase == 0 ? 0 : count * 2;
        uint32_t done = 0;
        uint32_t tile_pieces = 0;
        while (done < count) {
          const auto piece = t5qkv::flat_history_piece(
              tiling, offset + done, count - done, phase == 2);
          assert(done % 16 == 0 && piece.count > 0 && piece.count % 16 == 0);
          assert(piece.source_offset + piece.count <=
                 (piece.qkv ? 3 * 256 * 64 : 256 * 2048 * 64));
          for (uint32_t index = 0; index < piece.count; ++index) {
            const uint32_t destination = offset + done + index;
            const uint32_t row = destination / (2049 * 64);
            const uint32_t column = destination % (2049 * 64);
            const bool qkv = column >= 2048 * 64;
            const uint32_t expected =
                qkv ? row / 16 * 3 * 16 * 64 + (phase == 2 ? 2 : 1) * 16 * 64 +
                          row % 16 * 64 + column - 2048 * 64
                    : row * 2048 * 64 + column;
            assert(piece.qkv == qkv && piece.source_offset + index == expected);
            ++(phase == 2 ? value_seen : key_seen)[destination];
          }
          read_bytes += piece.qkv ? 0 : piece.count * 2;
          done += piece.count;
          ++pieces;
          ++tile_pieces;
        }
        assert(tile_pieces <= 3);
        if (rank + 1 < range.end) {
          physical = t5qkv::advance_dispersed_output(phase, physical);
        }
      }
    }
    assert(core_tasks == 29 || core_tasks == 30);
  }
  for (const auto& tensor : seen) {
    for (const uint32_t count : tensor) {
      assert(count == 1);
    }
  }
  for (const uint8_t count : key_seen) {
    assert(count == 1);
  }
  for (const uint8_t count : value_seen) {
    assert(count == 1);
  }
  assert(read_bytes == uint64_t{2} * 256 * 2048 * 64 * 2);
  assert(normal_write + bypass_write == uint64_t{2} * output_elements * 2);
  if (spaced) {
    assert(t5qkv::kDispersedNormalSlots == 13);
    assert(normal_write == 54706176 && bypass_write == 79577088);
  }
  assert(read_bytes + normal_write + 131072 < uint64_t{190} * 1024 * 1024);
  std::array<uint32_t, 13> words{};
  std::memcpy(words.data(), &tiling, sizeof(tiling));
  tiling.past_length = 2047;
  assert(!t5qkv::uses_flat_history(tiling));
  std::cout << "{\"scope\":\"host_dispersed_write_mapping_only\",\"npu_"
               "executed\":false,\"status\":\"passed\",\"tiles\":1426,"
               "\"elements_checked\":67141632,\"normal_write_slots\":"
            << t5qkv::kDispersedNormalSlots << ",\"pieces\":" << pieces
            << ",\"normal_history_read_bytes\":" << read_bytes
            << ",\"normal_key_write_bytes\":" << normal_write
            << ",\"bypass_write_bytes\":" << bypass_write
            << ",\"tiling_words\":[";
  for (uint32_t i = 0; i < words.size(); ++i) {
    std::cout << (i ? "," : "") << words[i];
  }
  std::cout << "]}\n";
}
