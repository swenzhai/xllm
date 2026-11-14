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
#include <vector>

#include "common/t5_split_qkv_append_cache_tiling.h"

int32_t main() {
  T5SplitQkvAppendCacheTilingData tiling{};
  assert(t5qkv::make_tiling(/*batch=*/16,
                            /*head_num=*/16,
                            /*past_length=*/2048,
                            /*head_dim=*/64,
                            /*available_vector_cores=*/48,
                            tiling));
  assert(t5qkv::uses_flat_history(tiling));
  const uint32_t output_elements = 256 * 2049 * 64;
  const uint32_t normal_elements = 240 * 2049 * 64;
  std::vector<uint8_t> key_seen(output_elements, 0);
  std::vector<uint8_t> value_seen(output_elements, 0);
  uint64_t gathered = 0;
  uint32_t tiles = 0;
  uint32_t pieces = 0;
  for (uint32_t phase = 0; phase < 3; ++phase) {
    const uint32_t begin = phase == 1 ? normal_elements : 0;
    const uint32_t end = phase == 0 ? normal_elements : output_elements;
    const uint32_t tasks =
        (end - begin + tiling.tile_elements - 1) / tiling.tile_elements;
    for (uint32_t core = 0; core < 48; ++core) {
      const auto range = t5qkv::partition_tasks(0, tasks, 48, core);
      for (uint32_t task = range.begin; task < range.end; ++task) {
        const uint32_t offset = begin + task * tiling.tile_elements;
        const uint32_t count = std::min(end - offset, tiling.tile_elements);
        assert(offset % 256 == 0 && count % 256 == 0 && count > 0);
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
          done += piece.count;
          gathered += piece.count;
          ++tile_pieces;
          ++pieces;
        }
        assert(tile_pieces <= 3 && done == count);
        ++tiles;
      }
    }
  }
  for (const uint8_t count : key_seen) {
    assert(count == 1);
  }
  for (const uint8_t count : value_seen) {
    assert(count == 1);
  }
  assert(gathered == uint64_t{2} * output_elements);
  std::array<uint32_t, 13> words{};
  std::memcpy(words.data(), &tiling, sizeof(tiling));
  tiling.past_length = 2047;
  assert(!t5qkv::uses_flat_history(tiling));
  std::cout << "{\"scope\":\"host_flat_mapping_only\",\"npu_executed\":false,"
               "\"status\":\"passed\",\"output_elements_checked\":"
            << gathered << ",\"tiles\":" << tiles << ",\"pieces\":" << pieces
            << ",\"tiling_words\":[";
  for (uint32_t i = 0; i < words.size(); ++i) {
    std::cout << (i ? "," : "") << words[i];
  }
  std::cout << "]}\n";
}
