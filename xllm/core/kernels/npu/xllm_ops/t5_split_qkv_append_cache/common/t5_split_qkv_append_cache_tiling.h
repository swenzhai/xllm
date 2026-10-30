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
  uint32_t cacheable_key_rows;
  uint32_t bypass_value_cache;
};
static_assert(sizeof(T5SplitQkvAppendCacheTilingData) == 13 * sizeof(uint32_t),
              "T5 QKV host/device tiling layout changed");

namespace t5qkv {
constexpr uint32_t kTileElements = 47104;
constexpr uint32_t kMaxVectorCores = 48;
constexpr uint64_t kMaxOutputElements = uint64_t{1} << 28;

#if defined(__CCE_AICORE__)
#define T5QKV_MAPPING_INLINE __aicore__ inline
#else
#define T5QKV_MAPPING_INLINE inline
#endif

constexpr uint64_t kL2BudgetBytes = uint64_t{172} * 1024 * 1024;
constexpr uint64_t kReadBalanceBudgetBytes = uint64_t{190} * 1024 * 1024;

struct CachePolicy {
  uint32_t key_elements;
  uint32_t key_tasks;
  bool bypass_value;
};

// Software admission budget, not a reservation or a residency guarantee.
T5QKV_MAPPING_INLINE CachePolicy
make_cache_policy(const T5SplitQkvAppendCacheTilingData& tiling) {
  const uint64_t n = uint64_t{tiling.batch} * tiling.head_num * tiling.head_dim;
  const uint64_t rows = uint64_t{tiling.batch} * tiling.head_num;
  const uint64_t row_elements =
      uint64_t{tiling.past_length + 1} * tiling.head_dim;
  if (8 * n * tiling.past_length + 12 * n <= kL2BudgetBytes) {
    return {static_cast<uint32_t>(rows * row_elements),
            tiling.cache_tasks_per_tensor,
            false};
  }
  const uint64_t base = 4 * n * (tiling.past_length + 2);
  const uint64_t remaining = kL2BudgetBytes > base ? kL2BudgetBytes - base : 0;
  const uint64_t candidate_rows = remaining / (2 * row_elements);
  const uint64_t admitted_rows = candidate_rows < rows ? candidate_rows : rows;
  return {static_cast<uint32_t>(admitted_rows * row_elements),
          static_cast<uint32_t>(admitted_rows * tiling.cache_tiles_per_head),
          true};
}

struct ReadBalancePolicy {
  bool enabled;
  uint32_t value_rows;
  uint32_t value_tasks;
};

// Retain Key input/output and admit only a bounded prefix of Value input.
// Value output uses the bypass hint. This budget is not a cache reservation.
T5QKV_MAPPING_INLINE ReadBalancePolicy
make_read_balance_policy(const T5SplitQkvAppendCacheTilingData& tiling) {
  const uint64_t n = uint64_t{tiling.batch} * tiling.head_num * tiling.head_dim;
  const uint64_t base = 2 * n * (2 * uint64_t{tiling.past_length} + 5);
  if (tiling.bypass_value_cache == 0 || tiling.past_length == 0 ||
      base >= kReadBalanceBudgetBytes) {
    return {};
  }
  const uint64_t rows = uint64_t{tiling.batch} * tiling.head_num;
  uint64_t cached_rows = (kReadBalanceBudgetBytes - base) /
                         (2 * uint64_t{tiling.past_length} * tiling.head_dim);
  cached_rows = cached_rows < rows ? cached_rows : rows;
  uint32_t divisor = tiling.cache_tiles_per_head;
  uint32_t cores = tiling.block_dim;
  while (cores != 0) {
    const uint32_t remainder = divisor % cores;
    divisor = cores;
    cores = remainder;
  }
  const uint32_t group = tiling.block_dim / divisor;
  if (cached_rows >= group) {
    cached_rows -= cached_rows % group;
  }
  return {true,
          static_cast<uint32_t>(cached_rows),
          static_cast<uint32_t>(cached_rows) * tiling.cache_tiles_per_head};
}

inline bool can_fuse_append(const T5SplitQkvAppendCacheTilingData& tiling) {
  if (tiling.past_length == 0) {
    return false;
  }
  const uint32_t history = tiling.past_length * tiling.head_dim;
  const uint32_t tail =
      history - (tiling.cache_tiles_per_head - 1) * tiling.tile_elements;
  const uint32_t padded_append = ((tiling.head_dim + 15) / 16) * 16;
  return tail % 16 == 0 && tail + padded_append <= tiling.tile_elements;
}

// make_tiling has already selected the task layout. Decode that decision
// without repeating tail division/modulo on every device copy task.
T5QKV_MAPPING_INLINE bool fuses_append(
    const T5SplitQkvAppendCacheTilingData& tiling) {
  return tiling.past_length != 0 &&
         tiling.total_tasks ==
             2 * tiling.cache_tasks_per_tensor + tiling.query_tasks;
}

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
  const uint32_t query_elements = result.head_num * result.head_dim;
  result.query_tiles_per_batch =
      (query_elements + kTileElements - 1) / kTileElements;
  result.cache_tasks_per_tensor =
      result.batch * result.head_num * result.cache_tiles_per_head;
  result.query_tasks = result.batch * result.query_tiles_per_batch;
  result.total_tasks =
      2 * result.cache_tasks_per_tensor +
      (can_fuse_append(result) ? 0 : 2 * result.batch * result.head_num) +
      result.query_tasks;
  const uint32_t cores = available_vector_cores < kMaxVectorCores
                             ? available_vector_cores
                             : kMaxVectorCores;
  result.block_dim = result.total_tasks < cores ? result.total_tasks : cores;
  const CachePolicy policy = make_cache_policy(result);
  result.cacheable_key_rows =
      policy.key_elements / ((result.past_length + 1) * result.head_dim);
  result.bypass_value_cache = policy.bypass_value ? 1 : 0;
  return true;
}

struct TaskRange {
  uint32_t begin;
  uint32_t end;
};

// Give each core a contiguous, balanced slice of one independent phase.
// History cache classes, fallback append and Q use this same assignment.
T5QKV_MAPPING_INLINE TaskRange partition_tasks(uint32_t first,
                                               uint32_t count,
                                               uint32_t block_dim,
                                               uint32_t core) {
  const uint32_t chunk = count / block_dim;
  const uint32_t extra = count % block_dim;
  const uint32_t begin = first + core * chunk + (core < extra ? core : extra);
  return {begin, begin + chunk + (core < extra ? 1 : 0)};
}

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
  const uint32_t rows = tiling.batch * tiling.head_num;
  const uint32_t append_task_end =
      cache_task_count + (fuses_append(tiling) ? 0 : 2 * rows);
  if (task_id >= append_task_end) {
    const uint32_t q_task = task_id - append_task_end;
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

  const uint32_t history_elements = tiling.past_length * tiling.head_dim;
  const uint32_t output_row_elements = history_elements + tiling.head_dim;
  if (task_id >= cache_task_count) {
    const uint32_t append_task = task_id - cache_task_count;
    const uint32_t cache_index = append_task / rows;
    const uint32_t row = append_task % rows;
    const uint32_t batch = row / tiling.head_num;
    const uint32_t head = row % tiling.head_num;
    task.output = cache_index == 0 ? Output::KEY : Output::VALUE;
    task.append_source_offset =
        batch * 3 * hd + (cache_index + 1) * hd + head * tiling.head_dim;
    task.append_destination_offset =
        row * output_row_elements + history_elements;
    task.append_count = tiling.head_dim;
    return task;
  }
  const uint32_t cache_index = task_id / tiling.cache_tasks_per_tensor;
  const uint32_t within_cache = task_id % tiling.cache_tasks_per_tensor;
  const uint32_t row = within_cache / tiling.cache_tiles_per_head;
  const uint32_t tile = within_cache % tiling.cache_tiles_per_head;
  const uint32_t tile_offset = tile * tiling.tile_elements;
  task.output = cache_index == 0 ? Output::KEY : Output::VALUE;
  task.source_offset = row * history_elements + tile_offset;
  task.destination_offset = row * output_row_elements + tile_offset;
  const uint32_t remaining = history_elements - tile_offset;
  task.count =
      remaining < tiling.tile_elements ? remaining : tiling.tile_elements;
  if (fuses_append(tiling) && tile + 1 == tiling.cache_tiles_per_head) {
    const uint32_t batch = row / tiling.head_num;
    const uint32_t head = row % tiling.head_num;
    task.append_source_offset =
        batch * 3 * hd + (cache_index + 1) * hd + head * tiling.head_dim;
    task.append_destination_offset = task.destination_offset + task.count;
    task.append_count = tiling.head_dim;
  }
  return task;
}
#undef T5QKV_MAPPING_INLINE
}  // namespace t5qkv
