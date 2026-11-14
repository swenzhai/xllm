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

#include "common/t5_split_qkv_append_cache_tiling.h"
#include "kernel_operator.h"

namespace t5qkv {
class KernelT5SplitQkvAppendCache final {
 public:
  __aicore__ inline void init(GM_ADDR qkv,
                              GM_ADDR past_key,
                              GM_ADDR past_value,
                              GM_ADDR query,
                              GM_ADDR key,
                              GM_ADDR value,
                              const T5SplitQkvAppendCacheTilingData& tiling) {
    tiling_ = tiling;
    const ReadBalancePolicy read_policy = make_read_balance_policy(tiling);
    balanced_ = read_policy.enabled;
    normal_value_tasks_ = read_policy.value_tasks;
    // Select the already supported Key-chain path before any DMA when a
    // supplied Key base would mix cache modes within one output cache line.
    if (uses_flat_history(tiling) &&
        reinterpret_cast<uint64_t>(key) % 512 != 0) {
      balanced_ = true;
      normal_value_tasks_ = 0;
    }
    flat_ = can_pair_history_sources(tiling,
                                     reinterpret_cast<uint64_t>(past_key),
                                     reinterpret_cast<uint64_t>(past_value),
                                     reinterpret_cast<uint64_t>(key),
                                     reinterpret_cast<uint64_t>(value));
    history_slots_ = flat_ ? 3 : 2;
    query_fused_ = flat_ && tiling.head_num <= 32;
    const uint32_t query_elements =
        tiling.batch * tiling.head_num * tiling.head_dim;
    if (flat_) {
      source_gap_bytes_ =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(past_value) -
                                reinterpret_cast<uint64_t>(past_key));
      past_pair_.SetGlobalBuffer(
          reinterpret_cast<__gm__ uint16_t*>(past_key),
          uint64_t{source_gap_bytes_} / 2 +
              uint64_t{query_elements} * tiling.past_length);
    }
    qkv_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(qkv),
                         3 * query_elements);
    if (tiling.past_length != 0) {
      past_key_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(past_key),
                                query_elements * tiling.past_length);
      past_value_bypass_.SetGlobalBuffer(
          reinterpret_cast<__gm__ uint16_t*>(past_value),
          query_elements * tiling.past_length);
      past_value_bypass_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
      past_value_.SetGlobalBuffer(
          reinterpret_cast<__gm__ uint16_t*>(past_value),
          query_elements * tiling.past_length);
    }
    query_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(query),
                           query_elements);
    key_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(key),
                         query_elements * (tiling.past_length + 1));
    // Both views name the same GM allocation. Selecting the view avoids
    // rebuilding the hint address and dividing by row size for every task.
    key_bypass_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(key),
                                query_elements * (tiling.past_length + 1));
    if (!balanced_) {
      key_bypass_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    value_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(value),
                           query_elements * (tiling.past_length + 1));
    // The serialized Key prefix supports the streaming plan. The balanced
    // Value-input quota is computed once, never inside the copy loop.
    cacheable_key_elements_ =
        tiling.cacheable_key_rows * (tiling.past_length + 1) * tiling.head_dim;
    cacheable_key_tasks_ =
        tiling.cacheable_key_rows * tiling.cache_tiles_per_head;
    if (tiling.bypass_value_cache != 0) {
      value_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    pipe_.InitBuffer(buffer_,
                     (flat_ ? 3 * kStreamingSlotElements +
                                  (query_fused_ ? kFusedQueryElements : 0)
                            : 2 * tiling.tile_elements) *
                         sizeof(uint16_t));
    for (uint32_t slot = 0; slot < history_slots_; ++slot) {
      read_events_[slot] = static_cast<int32_t>(
          pipe_.AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
      write_events_[slot] = static_cast<int32_t>(
          pipe_.AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }
    if (query_fused_) {
      read_events_[3] = static_cast<int32_t>(
          pipe_.AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
    }
  }

  __aicore__ inline void process() {
    const uint32_t history_tasks = 2 * tiling_.cache_tasks_per_tensor;
    const uint32_t append_tasks =
        fuses_append(tiling_) ? 0 : 2 * tiling_.batch * tiling_.head_num;
    const uint32_t core = AscendC::GetBlockIdx();
    if (flat_) {
      ranges_[0] = balanced_stream_range(core);
      ranges_[1] = {0, 0};
      ranges_[2] = {0, 0};
    } else if (balanced_) {
      ranges_[0] = partition_tasks(
          0, tiling_.cache_tasks_per_tensor, tiling_.block_dim, core);
      ranges_[1] =
          partition_tasks(0, normal_value_tasks_, tiling_.block_dim, core);
      ranges_[2] =
          partition_tasks(normal_value_tasks_,
                          tiling_.cache_tasks_per_tensor - normal_value_tasks_,
                          tiling_.block_dim,
                          core);
    } else {
      ranges_[0] =
          partition_tasks(0, cacheable_key_tasks_, tiling_.block_dim, core);
      ranges_[1] =
          partition_tasks(cacheable_key_tasks_,
                          tiling_.cache_tasks_per_tensor - cacheable_key_tasks_,
                          tiling_.block_dim,
                          core);
      ranges_[2] = partition_tasks(
          0, tiling_.cache_tasks_per_tensor, tiling_.block_dim, core);
    }
    if (query_fused_) {
      prefetch_query();
    }
    if (flat_) {
      process_streaming_history();
    } else {
      process_history_pipeline<false>();
    }
    process_phase(history_tasks, append_tasks);
    if (!query_fused_) {
      if (flat_) {
        const TaskRange query = balanced_query_range(tiling_, core);
        for (uint32_t task = query.begin; task < query.end; ++task) {
          process_task(history_tasks + append_tasks + task);
        }
      } else {
        process_phase(history_tasks + append_tasks, tiling_.query_tasks);
      }
    }
    for (uint32_t slot = 0; slot < history_slots_; ++slot) {
      pipe_.ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(read_events_[slot]);
      pipe_.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(write_events_[slot]);
    }
    if (query_fused_) {
      pipe_.ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(read_events_[3]);
    }
  }

 private:
  enum class HistoryOutput : uint32_t { KEY = 0, BYPASS_KEY = 1, VALUE = 2 };

  struct HistoryWrite {
    uint32_t destination_offset;
    uint32_t count;
    HistoryOutput output;
  };

  struct Cursor {
    uint32_t phase;
    uint32_t task;
    uint32_t row;
    uint32_t tile;
  };

  template <bool Flat>
  __aicore__ inline void enter_phase(Cursor& cursor) {
    while (cursor.phase < 3 &&
           ranges_[cursor.phase].begin == ranges_[cursor.phase].end) {
      ++cursor.phase;
    }
    if (cursor.phase < 3) {
      cursor.task = ranges_[cursor.phase].begin;
      if constexpr (!Flat) {
        cursor.row = cursor.task / tiling_.cache_tiles_per_head;
        cursor.tile = cursor.task % tiling_.cache_tiles_per_head;
      } else {
        cursor.row = AscendC::GetBlockIdx();
        cursor.tile = 0;
      }
    }
  }

  template <bool Flat>
  __aicore__ inline void advance(Cursor& cursor) {
    ++cursor.task;
    if (cursor.task == ranges_[cursor.phase].end) {
      ++cursor.phase;
      enter_phase<Flat>(cursor);
    } else if constexpr (!Flat) {
      ++cursor.tile;
      if (cursor.tile == tiling_.cache_tiles_per_head) {
        cursor.tile = 0;
        ++cursor.row;
      }
    } else {
      ++cursor.tile;
    }
  }

  __aicore__ inline void prefetch_query() {
    query_range_ = balanced_query_range(tiling_, AscendC::GetBlockIdx());
    if (query_range_.begin == query_range_.end) {
      return;
    }
    const uint32_t hd = tiling_.head_num * tiling_.head_dim;
    auto local = buffer_.Get<uint16_t>()[3 * kStreamingSlotElements];
    const AscendC::DataCopyExtParams copy{
        static_cast<uint16_t>(query_range_.end - query_range_.begin),
        hd * 2,
        hd * 4,
        0,
        0};
    const AscendC::DataCopyPadExtParams<uint16_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(local, qkv_[query_range_.begin * 3 * hd], copy, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(read_events_[3]);
  }
  __aicore__ inline void store_fused_query() {
    if (!query_fused_ || query_range_.begin == query_range_.end) {
      return;
    }
    const uint32_t hd = tiling_.head_num * tiling_.head_dim;
    auto local = buffer_.Get<uint16_t>()[3 * kStreamingSlotElements];
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(read_events_[3]);
    const AscendC::DataCopyExtParams copy{
        1, (query_range_.end - query_range_.begin) * hd * 2, 0, 0, 0};
    AscendC::DataCopyPad(query_[query_range_.begin * hd], local, copy);
    // Query storage is never reused. All later history stores and W flags are
    // in the same MTE3 FIFO, so final history-W drains also complete this
    // store.
  }

  __aicore__ inline HistoryWrite prefetch_flat(const Cursor& cursor,
                                               uint32_t slot) {
    const BalancedPairPiece block =
        balanced_pair_piece(cursor.row, cursor.tile);
    const uint32_t offset = block.offset;
    const uint32_t count = block.count;
    auto local = buffer_.Get<uint16_t>()[slot * kStreamingSlotElements];
    const AscendC::DataCopyPadExtParams<uint16_t> pad{false, 0, 0, 0};
    uint32_t done = 0;
    while (done < count) {
      const FlatHistoryPiece piece =
          flat_history_piece(tiling_, offset + done, count - done, false);
      const uint32_t bytes = piece.count * 2;
      const uint32_t pitch = piece.qkv ? tiling_.head_num * tiling_.head_dim * 2
                                       : source_gap_bytes_;
      const AscendC::DataCopyExtParams copy{
          2,
          bytes,
          pitch - bytes,
          (kPairedPlaneElements - piece.count) / 16,
          0};
      const auto& source = piece.qkv ? qkv_ : past_pair_;
      AscendC::DataCopyPad(local[done], source[piece.source_offset], copy, pad);
      done += piece.count;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(read_events_[slot]);
    return {offset,
            count,
            interleaved_key_normal(cursor.tile) ? HistoryOutput::KEY
                                                : HistoryOutput::BYPASS_KEY};
  }

  template <bool Flat>
  __aicore__ inline HistoryWrite prefetch(const Cursor& cursor, uint32_t slot) {
    if constexpr (Flat) {
      return prefetch_flat(cursor, slot);
    }
    const bool value = balanced_ ? cursor.phase != 0 : cursor.phase == 2;
    const auto& source =
        !value ? past_key_
               : (balanced_ && cursor.phase == 2 ? past_value_bypass_
                                                 : past_value_);
    const uint32_t history_elements = tiling_.past_length * tiling_.head_dim;
    const uint32_t tile_offset = cursor.tile * tiling_.tile_elements;
    const uint32_t remaining = history_elements - tile_offset;
    const uint32_t count =
        remaining < tiling_.tile_elements ? remaining : tiling_.tile_elements;
    const bool append = fuses_append(tiling_) &&
                        cursor.tile + 1 == tiling_.cache_tiles_per_head;
    auto local = buffer_.Get<uint16_t>()[slot * tiling_.tile_elements];
    const AscendC::DataCopyExtParams copy{1, count * 2, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<uint16_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(
        local, source[cursor.row * history_elements + tile_offset], copy, pad);
    if (append) {
      const uint32_t batch = cursor.row / tiling_.head_num;
      const uint32_t head = cursor.row % tiling_.head_num;
      const uint32_t hd = tiling_.head_num * tiling_.head_dim;
      const uint32_t append_offset =
          batch * 3 * hd + (value ? 2 : 1) * hd + head * tiling_.head_dim;
      const AscendC::DataCopyExtParams tail{1, tiling_.head_dim * 2, 0, 0, 0};
      AscendC::DataCopyPad(local[count], qkv_[append_offset], tail, pad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(read_events_[slot]);
    return {cursor.row * (history_elements + tiling_.head_dim) + tile_offset,
            count + (append ? tiling_.head_dim : 0),
            balanced_ ? (value ? HistoryOutput::VALUE : HistoryOutput::KEY)
                      : static_cast<HistoryOutput>(cursor.phase)};
  }

  __aicore__ inline void process_streaming_history() {
    const uint32_t total = ranges_[0].end - ranges_[0].begin;
    if (total == 0) {
      return;
    }
    Cursor cursor{};
    enter_phase<true>(cursor);
    HistoryWrite pending[3]{};
    const uint32_t initial = total < 3 ? total : 3;
    for (uint32_t slot = 0; slot < initial; ++slot) {
      pending[slot] = prefetch_flat(cursor, slot);
      advance<true>(cursor);
    }
    store_fused_query();
    for (uint32_t index = 0; index < total; ++index) {
      const uint32_t slot = index % 3;
      const HistoryWrite current = pending[slot];
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(read_events_[slot]);
      auto local = buffer_.Get<uint16_t>()[slot * kStreamingSlotElements];
      auto& destination =
          current.output == HistoryOutput::KEY
              ? key_
              : (current.output == HistoryOutput::BYPASS_KEY ? key_bypass_
                                                             : value_);
      const AscendC::DataCopyExtParams copy{1, current.count * 2, 0, 0, 0};
      AscendC::DataCopyPad(
          destination[current.destination_offset], local, copy);
      AscendC::DataCopyPad(value_[current.destination_offset],
                           local[kPairedPlaneElements],
                           copy);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(write_events_[slot]);
      if (index + 3 < total) {
        // Refill only after this slot's preceding store. The other two slots
        // retain prefetched data and their independent read/write event IDs.
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(write_events_[slot]);
        pending[slot] = prefetch_flat(cursor, slot);
        advance<true>(cursor);
      }
    }
    for (uint32_t slot = 0; slot < initial; ++slot) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(write_events_[slot]);
    }
  }

  template <bool Flat>
  __aicore__ inline void process_history_pipeline() {
    uint32_t total = 0;
    for (uint32_t phase = 0; phase < 3; ++phase) {
      total += ranges_[phase].end - ranges_[phase].begin;
    }
    if (total == 0) {
      return;
    }
    Cursor cursor{};
    enter_phase<Flat>(cursor);
    uint32_t slot = 0;
    // Each slot owns one event in each direction. Consume the prior store
    // before reusing its UB region, including at cache-class boundaries.
    bool used[2] = {true, false};
    HistoryWrite current = prefetch<Flat>(cursor, slot);
    advance<Flat>(cursor);
    for (uint32_t index = 0; index < total; ++index) {
      HistoryWrite next{};
      const uint32_t next_slot = slot ^ 1;
      if (index + 1 < total) {
        if (used[next_slot]) {
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
              write_events_[next_slot]);
        }
        next = prefetch<Flat>(cursor, next_slot);
        used[next_slot] = true;
        advance<Flat>(cursor);
      }
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(read_events_[slot]);
      auto local = buffer_.Get<uint16_t>()[slot * tiling_.tile_elements];
      auto& destination =
          current.output == HistoryOutput::KEY
              ? key_
              : (current.output == HistoryOutput::BYPASS_KEY ? key_bypass_
                                                             : value_);
      const AscendC::DataCopyExtParams copy{1, current.count * 2, 0, 0, 0};
      AscendC::DataCopyPad(
          destination[current.destination_offset], local, copy);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(write_events_[slot]);
      current = next;
      slot = next_slot;
    }
    for (uint32_t index = 0; index < 2; ++index) {
      if (used[index]) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(write_events_[index]);
      }
    }
  }

  __aicore__ inline void process_phase(uint32_t first, uint32_t count) {
    const TaskRange tasks = partition_tasks(
        first, count, tiling_.block_dim, AscendC::GetBlockIdx());
    for (uint32_t task_id = tasks.begin; task_id < tasks.end; ++task_id) {
      process_task(task_id);
    }
  }

  __aicore__ inline void process_task(uint32_t task_id) {
    const CopyTask task = decode_task(tiling_, task_id);
    if (task.output == Output::QUERY) {
      copy_segment(query_,
                   qkv_,
                   task.destination_offset,
                   task.source_offset,
                   task.count);
    } else if (task.output == Output::KEY) {
      const uint32_t offset = task.count > 0 ? task.destination_offset
                                             : task.append_destination_offset;
      auto& destination = offset < cacheable_key_elements_ ? key_ : key_bypass_;
      copy_segment(destination,
                   past_key_,
                   task.destination_offset,
                   task.source_offset,
                   task.count);
      copy_segment(destination,
                   qkv_,
                   task.append_destination_offset,
                   task.append_source_offset,
                   task.append_count);
    } else {
      copy_segment(value_,
                   past_value_,
                   task.destination_offset,
                   task.source_offset,
                   task.count);
      copy_segment(value_,
                   qkv_,
                   task.append_destination_offset,
                   task.append_source_offset,
                   task.append_count);
    }
  }

  __aicore__ inline void copy_segment(
      AscendC::GlobalTensor<uint16_t>& destination,
      const AscendC::GlobalTensor<uint16_t>& source,
      uint32_t destination_offset,
      uint32_t source_offset,
      uint32_t count) {
    if (count == 0) {
      return;
    }
    auto local = buffer_.Get<uint16_t>();
    const AscendC::DataCopyExtParams copy{1, count * 2, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<uint16_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(local, source[source_offset], copy, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(read_events_[0]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(read_events_[0]);
    AscendC::DataCopyPad(destination[destination_offset], local, copy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(write_events_[0]);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(write_events_[0]);
  }

  bool query_fused_ = false;
  TaskRange query_range_{};
  uint32_t source_gap_bytes_ = 0;
  uint32_t history_slots_ = 2;
  bool flat_ = false;
  bool balanced_ = false;
  uint32_t normal_value_tasks_ = 0;
  uint32_t cacheable_key_elements_ = 0;
  uint32_t cacheable_key_tasks_ = 0;
  T5SplitQkvAppendCacheTilingData tiling_{};
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
  int32_t read_events_[4]{};
  int32_t write_events_[3]{};
  TaskRange ranges_[3]{};
  AscendC::GlobalTensor<uint16_t> qkv_, past_key_, past_value_, query_, key_,
      key_bypass_, value_, past_value_bypass_, past_pair_;
};
}  // namespace t5qkv
