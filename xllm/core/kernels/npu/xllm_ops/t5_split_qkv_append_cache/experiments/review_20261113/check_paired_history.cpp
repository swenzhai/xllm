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

int32_t main() {
  T5SplitQkvAppendCacheTilingData tiling{};
  assert(t5qkv::make_tiling(/*batch=*/16,
                            /*head_num=*/16,
                            /*past_length=*/2048,
                            /*head_dim=*/64,
                            /*available_vector_cores=*/48,
                            tiling));
  constexpr uint64_t kBase = 0x100000000ULL;
  constexpr uint64_t kOutputKey = 0x300000000ULL;
  constexpr uint64_t kOutputValue = 0x400000000ULL;
  for (const uint64_t gap :
       {uint64_t{69206016}, uint64_t{67108866}, uint64_t{0xfffffffe}}) {
    assert(t5qkv::can_pair_history_sources(
        tiling, kBase, kBase + gap, kOutputKey, kOutputValue));
  }
  assert(t5qkv::can_pair_history_sources(
      tiling, kBase + 2, kBase + 69206018, kOutputKey, kOutputValue));
  assert(!t5qkv::can_pair_history_sources(
      tiling, kBase, kBase, kOutputKey, kOutputValue));
  assert(!t5qkv::can_pair_history_sources(
      tiling, kBase, kBase - 2, kOutputKey, kOutputValue));
  assert(!t5qkv::can_pair_history_sources(
      tiling, kBase, kBase + 47102, kOutputKey, kOutputValue));
  assert(!t5qkv::can_pair_history_sources(
      tiling, kBase, kBase + 0x100000000ULL, kOutputKey, kOutputValue));
  assert(!t5qkv::can_pair_history_sources(
      tiling, kBase + 1, kBase + 69206017, kOutputKey, kOutputValue));
  assert(!t5qkv::can_pair_history_sources(
      tiling, kBase, kBase + 69206016, kOutputKey + 2, kOutputValue));
  assert(!t5qkv::can_pair_history_sources(
      tiling, kBase, kBase + 69206016, kOutputKey, kOutputValue + 2));
  const uint32_t output_elements = 256 * 2049 * 64;
  std::array<uint32_t, 1426> seen{};
  std::vector<uint8_t> key_seen(output_elements, 0);
  std::vector<uint8_t> value_seen(output_elements, 0);
  uint64_t normal_write = 0;
  uint64_t bypass_write = 0;
  uint64_t history_read = 0;
  uint32_t reads = 0;
  uint32_t stores = 0;
  for (uint32_t core = 0; core < 48; ++core) {
    uint32_t core_tasks = 0;
    for (uint32_t phase = 0; phase < 2; ++phase) {
      const auto range = t5qkv::paired_output_range(tiling, phase, core);
      uint32_t physical = t5qkv::paired_output_task(phase, range.begin);
      for (uint32_t rank = range.begin; rank < range.end; ++rank) {
        assert(physical == t5qkv::paired_output_task(phase, rank) &&
               physical < seen.size());
        ++seen[physical];
        const uint32_t parent = physical / 2;
        const uint32_t lane = parent % 16;
        const bool normal =
            parent >= 704 || (lane != 4 && lane != 9 && lane != 14);
        assert(normal == (phase == 0));
        const uint32_t offset = physical * t5qkv::kPairedPlaneElements;
        const uint32_t count =
            std::min(output_elements - offset, t5qkv::kPairedPlaneElements);
        assert(offset % 256 == 0 && count % 256 == 0 && count > 0);
        normal_write += phase == 0 ? count * 2 : 0;
        bypass_write += count * 2 + (phase == 1 ? count * 2 : 0);
        stores += 2;
        uint32_t done = 0;
        while (done < count) {
          const auto piece = t5qkv::flat_history_piece(
              tiling, offset + done, count - done, false);
          assert(piece.count > 0 &&
                 piece.count <= t5qkv::kPairedPlaneElements &&
                 piece.count % 16 == 0 && done % 16 == 0);
          const uint32_t bytes = piece.count * 2;
          const uint32_t destination_stride =
              (t5qkv::kPairedPlaneElements - piece.count) / 16;
          const uint32_t local = (core_tasks % 2 * 47104 + done) * 2;
          const uint32_t second_local = local + bytes + destination_stride * 32;
          assert(second_local == local + t5qkv::kPairedPlaneElements * 2);
          assert(second_local + bytes <= (core_tasks % 2 + 1) * 47104 * 2);
          for (const uint64_t gap :
               {uint64_t{69206016}, uint64_t{67108866}, uint64_t{0xfffffffe}}) {
            const uint64_t pitch = piece.qkv ? 16 * 64 * 2 : gap;
            assert(pitch >= bytes && pitch - bytes <= 0xffffffffULL);
            const uint64_t first = kBase + piece.source_offset * 2;
            const uint64_t second = first + bytes + (pitch - bytes);
            const uint64_t wanted =
                piece.qkv ? kBase + (piece.source_offset + 16 * 64) * 2
                          : kBase + gap + piece.source_offset * 2;
            assert(second == wanted);
          }
          for (uint32_t i = 0; i < piece.count; ++i) {
            const uint32_t destination = offset + done + i;
            const uint32_t row = destination / (2049 * 64);
            const uint32_t column = destination % (2049 * 64);
            const bool qkv = column >= 2048 * 64;
            const uint32_t key_expected = qkv ? row / 16 * 3 * 16 * 64 +
                                                    16 * 64 + row % 16 * 64 +
                                                    column - 2048 * 64
                                              : row * 2048 * 64 + column;
            const uint32_t value_expected =
                qkv ? row / 16 * 3 * 16 * 64 + 2 * 16 * 64 + row % 16 * 64 +
                          column - 2048 * 64
                    : row * 2048 * 64 + column;
            assert(piece.qkv == qkv && piece.source_offset + i == key_expected);
            assert(piece.source_offset + i + (qkv ? 16 * 64 : 0) ==
                   value_expected);
            assert(key_expected < (qkv ? 3 * 256 * 64 : 256 * 2048 * 64));
            assert(value_expected < (qkv ? 3 * 256 * 64 : 256 * 2048 * 64));
            ++key_seen[destination];
            ++value_seen[destination];
          }
          history_read += piece.qkv ? 0 : piece.count * 4;
          done += piece.count;
          ++reads;
        }
        ++core_tasks;
        if (rank + 1 < range.end) {
          physical = t5qkv::advance_paired_output(phase, physical);
        }
      }
    }
    assert(core_tasks == 29 || core_tasks == 30);
    assert(t5qkv::paired_output_range(tiling, 2, core).begin ==
           t5qkv::paired_output_range(tiling, 2, core).end);
  }
  for (const uint32_t count : seen) {
    assert(count == 1);
  }
  for (const uint8_t count : key_seen) {
    assert(count == 1);
  }
  for (const uint8_t count : value_seen) {
    assert(count == 1);
  }
  assert(normal_write == 54706176 && bypass_write == 79577088 &&
         history_read == 134217728);
  std::array<uint32_t, 13> words{};
  std::memcpy(words.data(), &tiling, sizeof(tiling));
  tiling.past_length = 2047;
  assert(!t5qkv::can_pair_history_sources(
      tiling, kBase, kBase + 69206016, kOutputKey, kOutputValue));
  std::cout << "{\"scope\":\"paired_2d_CPU_mapping_only\",\"npu_executed\":"
               "false,\"status\":\"passed\",\"paired_tasks\":1426,\"elements_"
               "checked\":67141632,\"read_DMA_instructions\":"
            << reads << ",\"write_DMA_instructions\":" << stores
            << ",\"normal_key_write_bytes\":" << normal_write
            << ",\"bypass_write_bytes\":" << bypass_write
            << ",\"normal_history_read_bytes\":" << history_read
            << ",\"aligned_store_variant\":"
            << (t5qkv::kPairedAlignedStores ? "true" : "false")
            << ",\"tiling_words\":[";
  for (uint32_t i = 0; i < words.size(); ++i) {
    std::cout << (i ? "," : "") << words[i];
  }
  std::cout << "]}\n";
}
