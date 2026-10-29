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

#pragma once

#include <cstdint>

// Shared byte-for-byte by the ACLNN host tiler and the direct-launch path.
struct T5SplitQkvAppendCacheTilingData {
  uint32_t batch;
  uint32_t head_num;
  uint32_t past_length;
  uint32_t head_dim;
  uint32_t tile_elements;
  uint32_t cache_tiles_per_head;
  uint32_t query_tiles_per_batch;
  uint32_t cache_tasks_per_tensor;
  uint32_t query_tasks;
  uint32_t total_tasks;
  uint32_t block_dim;
};
static_assert(sizeof(T5SplitQkvAppendCacheTilingData) == 11 * sizeof(uint32_t),
              "T5 QKV host/device tiling layout changed");

namespace t5qkv {
constexpr uint32_t kTileElements = 16384;
constexpr uint32_t kMaxVectorCores = 48;
constexpr uint64_t kMaxOutputElements = uint64_t{1} << 28;

// Shape/dtype/layout validation belongs to each public host entry. This helper
// also rejects invalid dimensions so direct and ACLNN tilers share the limits.
inline bool make_tiling(int64_t batch,
                        int64_t head_num,
                        int64_t past_length,
                        int64_t head_dim,
                        uint32_t available_vector_cores,
                        T5SplitQkvAppendCacheTilingData& result) {
  result = {};
  if (batch < 1 || batch > 64 || head_num < 1 || head_num > 128 ||
      past_length < 0 || past_length > 8192 || head_dim < 1 || head_dim > 256 ||
      available_vector_cores == 0) {
    return false;
  }
  const uint64_t outputs = static_cast<uint64_t>(batch) * head_num * head_dim *
                           (2 * static_cast<uint64_t>(past_length) + 3);
  if (outputs > kMaxOutputElements) {
    return false;
  }
  result.batch = static_cast<uint32_t>(batch);
  result.head_num = static_cast<uint32_t>(head_num);
  result.past_length = static_cast<uint32_t>(past_length);
  result.head_dim = static_cast<uint32_t>(head_dim);
  result.tile_elements = kTileElements;
  const uint32_t history_elements = result.past_length * result.head_dim;
  result.cache_tiles_per_head =
      (history_elements + kTileElements - 1) / kTileElements;
  // Empty histories retain one task per head solely to append the token.
  if (result.cache_tiles_per_head == 0) {
    result.cache_tiles_per_head = 1;
  }
  const uint32_t query_elements = result.head_num * result.head_dim;
  result.query_tiles_per_batch =
      (query_elements + kTileElements - 1) / kTileElements;
  result.cache_tasks_per_tensor =
      result.batch * result.head_num * result.cache_tiles_per_head;
  result.query_tasks = result.batch * result.query_tiles_per_batch;
  result.total_tasks = 2 * result.cache_tasks_per_tensor + result.query_tasks;
  const uint32_t cores = available_vector_cores < kMaxVectorCores
                             ? available_vector_cores
                             : kMaxVectorCores;
  result.block_dim = result.total_tasks < cores ? result.total_tasks : cores;
  return true;
}

#if defined(__CCE_AICORE__)
#define T5QKV_MAPPING_INLINE __aicore__ inline
#else
#define T5QKV_MAPPING_INLINE inline
#endif

enum class Output : uint32_t { QUERY = 0, KEY = 1, VALUE = 2 };

struct CopyTask {
  Output output;
  uint32_t source_offset;
  uint32_t destination_offset;
  uint32_t count;
  uint32_t append_source_offset;
  uint32_t append_destination_offset;
  uint32_t append_count;
};

// All offsets/counts are uint16 elements relative to the tensor's data pointer,
// which already includes any framework storage offset. Caller supplies a valid
// task_id in [0, total_tasks), and make_tiling bounds all products to uint32_t.
T5QKV_MAPPING_INLINE CopyTask
decode_task(const T5SplitQkvAppendCacheTilingData& tiling, uint32_t task_id) {
  CopyTask task{};
  const uint32_t hd = tiling.head_num * tiling.head_dim;
  const uint32_t cache_task_count = 2 * tiling.cache_tasks_per_tensor;
  if (task_id >= cache_task_count) {
    const uint32_t q_task = task_id - cache_task_count;
    const uint32_t batch = q_task / tiling.query_tiles_per_batch;
    const uint32_t tile_offset =
        (q_task % tiling.query_tiles_per_batch) * tiling.tile_elements;
    task.output = Output::QUERY;
    task.source_offset = batch * 3 * hd + tile_offset;
    task.destination_offset = batch * hd + tile_offset;
    const uint32_t remaining = hd - tile_offset;
    task.count =
        remaining < tiling.tile_elements ? remaining : tiling.tile_elements;
    return task;
  }

  const uint32_t cache_index = task_id / tiling.cache_tasks_per_tensor;
  const uint32_t within_cache = task_id % tiling.cache_tasks_per_tensor;
  const uint32_t row = within_cache / tiling.cache_tiles_per_head;
  const uint32_t tile = within_cache % tiling.cache_tiles_per_head;
  const uint32_t history_elements = tiling.past_length * tiling.head_dim;
  const uint32_t output_row_elements = history_elements + tiling.head_dim;
  const uint32_t tile_offset = tile * tiling.tile_elements;
  task.output = cache_index == 0 ? Output::KEY : Output::VALUE;
  task.source_offset = row * history_elements + tile_offset;
  task.destination_offset = row * output_row_elements + tile_offset;
  const uint32_t remaining = history_elements - tile_offset;
  task.count =
      remaining < tiling.tile_elements ? remaining : tiling.tile_elements;
  if (tile + 1 == tiling.cache_tiles_per_head) {
    const uint32_t batch = row / tiling.head_num;
    const uint32_t head = row % tiling.head_num;
    task.append_source_offset =
        batch * 3 * hd + (cache_index + 1) * hd + head * tiling.head_dim;
    task.append_destination_offset =
        row * output_row_elements + history_elements;
    task.append_count = tiling.head_dim;
  }
  return task;
}
#undef T5QKV_MAPPING_INLINE
}  // namespace t5qkv
