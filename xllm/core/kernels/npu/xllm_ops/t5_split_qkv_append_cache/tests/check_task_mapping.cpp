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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/t5_split_qkv_append_cache_tiling.h"

namespace {
using Interval = std::pair<uint32_t, uint32_t>;
using Source = std::pair<uint32_t, uint32_t>;

Source expected_source(const T5SplitQkvAppendCacheTilingData& tiling,
                       t5qkv::Output output,
                       uint32_t destination) {
  const uint32_t hd = tiling.head_num * tiling.head_dim;
  if (output == t5qkv::Output::QUERY) {
    return {0, (destination / hd) * 3 * hd + destination % hd};
  }
  const uint32_t history = tiling.past_length * tiling.head_dim;
  const uint32_t row_length = history + tiling.head_dim;
  const uint32_t row = destination / row_length;
  const uint32_t in_row = destination % row_length;
  const uint32_t cache_index = static_cast<uint32_t>(output);
  if (in_row < history) {
    return {cache_index, row * history + in_row};
  }
  return {0,
          (row / tiling.head_num) * 3 * hd + cache_index * hd +
              (row % tiling.head_num) * tiling.head_dim + in_row - history};
}

void check_segment(const T5SplitQkvAppendCacheTilingData& tiling,
                   const std::array<uint32_t, 3>& input_sizes,
                   const std::array<uint32_t, 3>& output_sizes,
                   std::array<std::vector<Interval>, 3>& intervals,
                   t5qkv::Output output,
                   uint32_t source,
                   uint32_t source_offset,
                   uint32_t destination_offset,
                   uint32_t count,
                   uint32_t storage_offset) {
  if (count == 0) {
    return;
  }
  const uint32_t output_index = static_cast<uint32_t>(output);
  assert(count <= tiling.tile_elements);
  assert(source_offset <= input_sizes[source]);
  assert(count <= input_sizes[source] - source_offset);
  assert(destination_offset <= output_sizes[output_index]);
  assert(count <= output_sizes[output_index] - destination_offset);
  // Mapping is affine inside each segment. Endpoints catch crossing a head,
  // batch, or history/append boundary with the wrong source tensor/stride.
  for (const uint32_t delta : {uint32_t{0}, count - 1}) {
    const Source expected =
        expected_source(tiling, output, destination_offset + delta);
    assert(source == expected.first);
    assert(storage_offset + source_offset + delta ==
           storage_offset + expected.second);
  }
  intervals[output_index].emplace_back(destination_offset,
                                       destination_offset + count);
}

uint32_t check_shape(int64_t batch,
                     int64_t heads,
                     int64_t length,
                     int64_t dim,
                     uint32_t core_count,
                     uint32_t storage_offset) {
  T5SplitQkvAppendCacheTilingData tiling{};
  assert(t5qkv::make_tiling(batch, heads, length, dim, core_count, tiling));
  assert(tiling.block_dim > 0);
  assert(tiling.block_dim <= 48 && tiling.block_dim <= core_count);
  assert(tiling.tile_elements == 47104);
  assert((tiling.tile_elements * sizeof(uint16_t)) % 32 == 0);
  const uint32_t query = static_cast<uint32_t>(batch * heads * dim);
  const uint32_t history = query * static_cast<uint32_t>(length);
  const uint32_t cache = query * static_cast<uint32_t>(length + 1);
  const std::array<uint32_t, 3> input_sizes{3 * query, history, history};
  const std::array<uint32_t, 3> output_sizes{query, cache, cache};
  std::array<std::vector<Interval>, 3> intervals;
  for (auto& output : intervals) {
    output.reserve(tiling.total_tasks * 2);
  }
  std::vector<uint8_t> visits(tiling.total_tasks, 0);
  uint32_t history_segments = 0;
  uint32_t append_segments = 0;
  const std::array<uint32_t, 3> phase_counts{
      2 * tiling.cache_tasks_per_tensor,
      t5qkv::fuses_append(tiling) ? 0 : 2 * tiling.batch * tiling.head_num,
      tiling.query_tasks};
  uint32_t first = 0;
  for (const uint32_t count : phase_counts) {
    uint32_t minimum = count;
    uint32_t maximum = 0;
    for (uint32_t core = 0; core < tiling.block_dim; ++core) {
      const t5qkv::TaskRange tasks =
          t5qkv::partition_tasks(first, count, tiling.block_dim, core);
      assert(tasks.begin >= first && tasks.end <= first + count);
      assert(tasks.begin <= tasks.end);
      minimum = std::min(minimum, tasks.end - tasks.begin);
      maximum = std::max(maximum, tasks.end - tasks.begin);
      for (uint32_t task_id = tasks.begin; task_id < tasks.end; ++task_id) {
        assert(visits[task_id]++ == 0);
        const t5qkv::CopyTask task = t5qkv::decode_task(tiling, task_id);
        // Fused tails carry adjacent, separately checked history and append
        // segments.
        assert(task.count > 0 || task.append_count > 0);
        if (task.count > 0 && task.append_count > 0) {
          assert(task.destination_offset + task.count ==
                 task.append_destination_offset);
          assert(task.count % 16 == 0);
          assert(task.count + ((task.append_count + 15) / 16) * 16 <=
                 tiling.tile_elements);
        }
        const uint32_t primary_source = static_cast<uint32_t>(task.output);
        check_segment(tiling,
                      input_sizes,
                      output_sizes,
                      intervals,
                      task.output,
                      primary_source,
                      task.source_offset,
                      task.destination_offset,
                      task.count,
                      storage_offset);
        check_segment(tiling,
                      input_sizes,
                      output_sizes,
                      intervals,
                      task.output,
                      /*source=*/0,
                      task.append_source_offset,
                      task.append_destination_offset,
                      task.append_count,
                      storage_offset);
        history_segments +=
            (task.output != t5qkv::Output::QUERY && task.count > 0);
        append_segments += (task.append_count > 0);
      }
    }
    assert(maximum - minimum <= 1);
    first += count;
  }
  assert(first == tiling.total_tasks);
  assert(std::all_of(
      visits.begin(), visits.end(), [](uint8_t count) { return count == 1; }));
  assert(append_segments == static_cast<uint32_t>(2 * batch * heads));
  assert(length != 0 || history_segments == 0);
  for (uint32_t output_index = 0; output_index < intervals.size();
       ++output_index) {
    auto& output = intervals[output_index];
    std::sort(output.begin(), output.end());
    uint32_t end = 0;
    for (const auto& interval : output) {
      assert(interval.first == end);  // Neither gaps nor overlapping writes.
      end = interval.second;
    }
    assert(end == output_sizes[output_index]);
  }
  return tiling.total_tasks;
}

void check_rejections() {
  T5SplitQkvAppendCacheTilingData tiling{};
  assert(!t5qkv::make_tiling(0, 1, 0, 1, 48, tiling));
  assert(!t5qkv::make_tiling(65, 1, 0, 1, 48, tiling));
  assert(!t5qkv::make_tiling(1, 129, 0, 1, 48, tiling));
  assert(!t5qkv::make_tiling(1, 1, -1, 1, 48, tiling));
  assert(!t5qkv::make_tiling(1, 1, 8193, 1, 48, tiling));
  assert(!t5qkv::make_tiling(1, 1, 0, 257, 48, tiling));
  assert(!t5qkv::make_tiling(64, 128, 8192, 256, 48, tiling));
  assert(!t5qkv::make_tiling(1, 1, 0, 1, 0, tiling));
  assert(tiling.total_tasks == 0);
}
}  // namespace

int32_t main() {
  static_assert(
      std::is_standard_layout<T5SplitQkvAppendCacheTilingData>::value);
  static_assert(std::is_trivial<T5SplitQkvAppendCacheTilingData>::value);
  const std::array<std::array<int64_t, 4>, 10> shapes{{{1, 1, 0, 1},
                                                       {4, 8, 0, 64},
                                                       {2, 3, 2, 5},
                                                       {2, 3, 127, 65},
                                                       {1, 1, 8192, 128},
                                                       {2, 1, 4097, 3},
                                                       {64, 128, 62, 256},
                                                       {1, 128, 1, 1},
                                                       {16, 16, 2048, 64},
                                                       {4, 128, 1024, 65}}};
  uint32_t checks = 0;
  uint32_t tasks = 0;
  for (const auto& shape : shapes) {
    for (const uint32_t core_count :
         {uint32_t{1}, uint32_t{7}, uint32_t{48}, uint32_t{96}}) {
      for (const uint32_t offset : {uint32_t{0}, uint32_t{1}}) {
        tasks += check_shape(
            shape[0], shape[1], shape[2], shape[3], core_count, offset);
        ++checks;
      }
    }
  }
  check_rejections();
  std::cout << "{\"scope\":\"host_address_mapping_only\",\"status\":\"passed\","
            << "\"npu_executed\":false,\"shape_core_offset_checks\":" << checks
            << ",\"decoded_tasks\":" << tasks << "}\n";
  return 0;
}
