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

constexpr uint64_t kL2BudgetBytes = uint64_t{190} * 1024 * 1024;
constexpr uint64_t kReadBalanceBudgetBytes = uint64_t{190} * 1024 * 1024;

struct CachePolicy {
  uint32_t key_elements;
  uint32_t key_tasks;
  bool bypass_value;
};

T5QKV_MAPPING_INLINE uint32_t cache_policy_gcd(uint32_t first,
                                               uint32_t second) {
  while (second != 0) {
    const uint32_t remainder = first % second;
    first = second;
    second = remainder;
  }
  return first;
}

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
  const uint64_t limited_rows = candidate_rows < rows ? candidate_rows : rows;
  // Balance each history cache class and keep its output boundary 512B aligned
  // relative to the fresh allocator-aligned output base.
  const uint32_t core_group =
      tiling.block_dim /
      cache_policy_gcd(tiling.cache_tiles_per_head, tiling.block_dim);
  const uint32_t alignment_group =
      256 / cache_policy_gcd(static_cast<uint32_t>(row_elements % 256),
                             /*second=*/256);
  const uint32_t row_group = core_group * alignment_group /
                             cache_policy_gcd(core_group, alignment_group);
  const uint64_t admitted_rows = limited_rows - limited_rows % row_group;
  return {static_cast<uint32_t>(admitted_rows * row_elements),
          static_cast<uint32_t>(admitted_rows * tiling.cache_tiles_per_head),
          true};
}

struct ReadBalancePolicy {
  bool enabled;
  uint32_t value_rows;
  uint32_t value_tasks;
};

// Experimental input-priority plan: retain all history reads and spend the
// remaining budget on a bounded Key-output prefix. Use the existing balanced
// cached-Key / bypass-Key / Value phase scheduler; Value output still bypasses.
T5QKV_MAPPING_INLINE ReadBalancePolicy
make_read_balance_policy(const T5SplitQkvAppendCacheTilingData& /*tiling*/) {
  return {};
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
// Flat output tiles coalesce row seams without padding or extra GM storage.
struct FlatHistoryPiece {
  uint32_t source_offset;
  uint32_t count;
  bool qkv;
};

T5QKV_MAPPING_INLINE bool uses_flat_history(
    const T5SplitQkvAppendCacheTilingData& tiling) {
  return tiling.batch * tiling.head_num == 256 && tiling.past_length == 2048 &&
         tiling.head_dim == 64 && tiling.block_dim == 48 &&
         tiling.tile_elements == 47104 && tiling.cacheable_key_rows == 240 &&
         tiling.bypass_value_cache != 0 && fuses_append(tiling) &&
         !make_read_balance_policy(tiling).enabled;
}

T5QKV_MAPPING_INLINE FlatHistoryPiece
flat_history_piece(const T5SplitQkvAppendCacheTilingData& tiling,
                   uint32_t output_offset,
                   uint32_t remaining,
                   bool value) {
  const uint32_t history = tiling.past_length * tiling.head_dim;
  const uint32_t row_elements = history + tiling.head_dim;
  const uint32_t row = output_offset / row_elements;
  const uint32_t column = output_offset % row_elements;
  if (column < history) {
    const uint32_t count =
        history - column < remaining ? history - column : remaining;
    return {row * history + column, count, false};
  }
  const uint32_t hd = tiling.head_num * tiling.head_dim;
  const uint32_t count =
      row_elements - column < remaining ? row_elements - column : remaining;
  return {row / tiling.head_num * 3 * hd + (value ? 2 : 1) * hd +
              row % tiling.head_num * tiling.head_dim + column - history,
          count,
          true};
}
constexpr uint32_t kStreamingTileElements = 31744;
constexpr uint32_t kPairedPlaneElements = kStreamingTileElements / 2;
constexpr bool kPairedAlignedStores = false;
constexpr uint32_t kStreamingSlotElements = kStreamingTileElements;
constexpr uint32_t kDispersedNormalSlots = 13;

T5QKV_MAPPING_INLINE uint32_t
dispersed_output_count(const T5SplitQkvAppendCacheTilingData& tiling) {
  const uint32_t elements = tiling.batch * tiling.head_num *
                            (tiling.past_length + 1) * tiling.head_dim;
  return (elements + kPairedPlaneElements - 1) / kPairedPlaneElements;
}

T5QKV_MAPPING_INLINE TaskRange
dispersed_output_range(const T5SplitQkvAppendCacheTilingData& tiling,
                       uint32_t phase,
                       uint32_t core) {
  const uint32_t total = dispersed_output_count(tiling);
  const uint32_t remainder = total % 16;
  const uint32_t normal =
      total / 16 * kDispersedNormalSlots +
      (remainder < kDispersedNormalSlots ? remainder : kDispersedNormalSlots);
  if (phase == 0) {
    return partition_tasks(0, normal, tiling.block_dim, core);
  }
  if (phase == 1) {
    const uint32_t rotated =
        (core + tiling.block_dim - normal % tiling.block_dim) %
        tiling.block_dim;
    return partition_tasks(0, total - normal, tiling.block_dim, rotated);
  }
  const uint32_t rotated =
      (core + tiling.block_dim - total % tiling.block_dim) % tiling.block_dim;
  return partition_tasks(0, total, tiling.block_dim, rotated);
}

// Keep the final partial period normal so admitted/bypassed bytes exactly
// match C21; distribute the three bypass slots within each full period.
constexpr uint32_t kSpacedFullPeriods =
    ((256 * 2049 * 64 + kStreamingTileElements - 1) / kStreamingTileElements) /
    16;
T5QKV_MAPPING_INLINE uint32_t dispersed_output_task(uint32_t phase,
                                                    uint32_t dense) {
  if (phase == 0) {
    if (dense >= kSpacedFullPeriods * 13) {
      return kSpacedFullPeriods * 16 + dense - kSpacedFullPeriods * 13;
    }
    const uint32_t lane = dense % 13;
    return dense / 13 * 16 + lane + (lane >= 4) + (lane >= 8) + (lane >= 12);
  }
  if (phase == 1) {
    return dense / 3 * 16 + 4 + dense % 3 * 5;
  }
  return dense;
}

T5QKV_MAPPING_INLINE uint32_t advance_dispersed_output(uint32_t phase,
                                                       uint32_t physical) {
  if (phase == 0) {
    if (physical >= kSpacedFullPeriods * 16) {
      return physical + 1;
    }
    ++physical;
    const uint32_t lane = physical % 16;
    return physical + (lane == 4 || lane == 9 || lane == 14);
  }
  if (phase == 1) {
    physical += 5;
    return physical + (physical % 16 == 3);
  }
  return physical + 1;
}
T5QKV_MAPPING_INLINE TaskRange
interleaved_output_range(const T5SplitQkvAppendCacheTilingData& tiling,
                         uint32_t core) {
  const TaskRange pairs = partition_tasks(
      0, dispersed_output_count(tiling), tiling.block_dim, core);
  return {2 * pairs.begin, 2 * pairs.end};
}

T5QKV_MAPPING_INLINE bool interleaved_key_normal(uint32_t physical) {
  return physical % 2 == 0;
}
T5QKV_MAPPING_INLINE bool interleaved_key_input_normal(uint32_t physical) {
  return !interleaved_key_normal(physical);
}
T5QKV_MAPPING_INLINE TaskRange
interleaved_query_range(const T5SplitQkvAppendCacheTilingData& tiling,
                        uint32_t core) {
  const uint32_t first_short =
      dispersed_output_count(tiling) % tiling.block_dim;
  if (first_short == 0) {
    return partition_tasks(0, tiling.query_tasks, tiling.block_dim, core);
  }
  if (core < first_short) {
    return {0, 0};
  }
  return partition_tasks(0,
                         tiling.query_tasks,
                         tiling.block_dim - first_short,
                         core - first_short);
}
T5QKV_MAPPING_INLINE bool can_pair_history_sources(
    const T5SplitQkvAppendCacheTilingData& tiling,
    uint64_t past_key,
    uint64_t past_value,
    uint64_t key,
    uint64_t value) {
  if (!uses_flat_history(tiling) || past_key == 0 || key == 0 || value == 0 ||
      key % 512 != 0 || value % 512 != 0 || past_value <= past_key ||
      ((past_key | past_value) & 1) != 0) {
    return false;
  }
  const uint64_t gap = past_value - past_key;
  return gap >= kPairedPlaneElements * 2 && gap <= 0xffffffffULL;
}
T5QKV_MAPPING_INLINE TaskRange
paired_stream_range(const T5SplitQkvAppendCacheTilingData& tiling,
                    uint32_t core) {
  return partition_tasks(
      0, dispersed_output_count(tiling), tiling.block_dim, core);
}
constexpr uint32_t kBalancedCoreElements = 699392;
constexpr uint32_t kBalancedFullFrames = 44;
constexpr uint32_t kBalancedCoreTasks = 46;
constexpr uint32_t kFusedQueryElements = 2048;
struct BalancedPairPiece {
  uint32_t offset;
  uint32_t count;
};
T5QKV_MAPPING_INLINE BalancedPairPiece balanced_pair_piece(uint32_t core,
                                                           uint32_t lane) {
  if (lane < kBalancedFullFrames) {
    return {core * kBalancedCoreElements + lane * kPairedPlaneElements,
            kPairedPlaneElements};
  }
  return {core * kBalancedCoreElements +
              kBalancedFullFrames * kPairedPlaneElements +
              (lane - kBalancedFullFrames) * 512,
          512};
}
T5QKV_MAPPING_INLINE TaskRange balanced_stream_range(uint32_t core) {
  return {core * kBalancedCoreTasks, (core + 1) * kBalancedCoreTasks};
}
T5QKV_MAPPING_INLINE TaskRange
balanced_query_range(const T5SplitQkvAppendCacheTilingData& tiling,
                     uint32_t core) {
  return partition_tasks(0, tiling.query_tasks, tiling.block_dim, core);
}
#undef T5QKV_MAPPING_INLINE
}  // namespace t5qkv
