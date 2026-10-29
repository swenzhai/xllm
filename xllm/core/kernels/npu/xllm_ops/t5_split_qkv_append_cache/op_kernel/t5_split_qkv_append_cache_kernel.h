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
    const uint32_t query_elements =
        tiling.batch * tiling.head_num * tiling.head_dim;
    qkv_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(qkv),
                         3 * query_elements);
    if (tiling.past_length != 0) {
      past_key_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(past_key),
                                query_elements * tiling.past_length);
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
    key_bypass_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    value_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(value),
                           query_elements * (tiling.past_length + 1));
    const CachePolicy policy = make_cache_policy(tiling);
    cacheable_key_elements_ = policy.key_elements;
    cacheable_key_tasks_ = policy.key_tasks;
    if (policy.bypass_value) {
      value_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    pipe_.InitBuffer(
        queue_, /*num=*/2, tiling.tile_elements * sizeof(uint16_t));
  }

  __aicore__ inline void process() {
    // Balance cache-admitted and bypassed history separately so every core
    // receives HBM traffic while retaining contiguous input-page locality.
    const uint32_t history_tasks = 2 * tiling_.cache_tasks_per_tensor;
    const uint32_t append_tasks =
        fuses_append(tiling_) ? 0 : 2 * tiling_.batch * tiling_.head_num;
    process_history(0, cacheable_key_tasks_);
    process_history(cacheable_key_tasks_,
                    tiling_.cache_tasks_per_tensor - cacheable_key_tasks_);
    process_history(tiling_.cache_tasks_per_tensor,
                    tiling_.cache_tasks_per_tensor);
    process_phase(history_tasks, append_tasks);
    process_phase(history_tasks + append_tasks, tiling_.query_tasks);
  }

 private:
  __aicore__ inline void prefetch_history(uint32_t task_id) {
    const CopyTask task = decode_task(tiling_, task_id);
    auto local = queue_.AllocTensor<uint16_t>();
    const AscendC::DataCopyExtParams copy{
        1, task.count * static_cast<uint32_t>(sizeof(uint16_t)), 0, 0, 0};
    const AscendC::DataCopyPadExtParams<uint16_t> pad{false, 0, 0, 0};
    const auto& source = task.output == Output::KEY ? past_key_ : past_value_;
    AscendC::DataCopyPad(local, source[task.source_offset], copy, pad);
    if (task.append_count != 0) {
      const AscendC::DataCopyExtParams append_copy{
          1,
          task.append_count * static_cast<uint32_t>(sizeof(uint16_t)),
          0,
          0,
          0};
      AscendC::DataCopyPad(
          local[task.count], qkv_[task.append_source_offset], append_copy, pad);
    }
    queue_.EnQue(local);
  }

  __aicore__ inline void process_history(uint32_t first, uint32_t count) {
    const TaskRange tasks = partition_tasks(
        first, count, tiling_.block_dim, AscendC::GetBlockIdx());
    if (tasks.begin == tasks.end) {
      return;
    }
    prefetch_history(tasks.begin);
    for (uint32_t task_id = tasks.begin; task_id < tasks.end; ++task_id) {
      if (task_id + 1 < tasks.end) {
        prefetch_history(task_id + 1);
      }
      const CopyTask task = decode_task(tiling_, task_id);
      auto local = queue_.DeQue<uint16_t>();
      const AscendC::DataCopyExtParams copy{
          1,
          (task.count + task.append_count) *
              static_cast<uint32_t>(sizeof(uint16_t)),
          0,
          0,
          0};
      auto& destination =
          task.output == Output::VALUE
              ? value_
              : (task.destination_offset < cacheable_key_elements_
                     ? key_
                     : key_bypass_);
      AscendC::DataCopyPad(destination[task.destination_offset], local, copy);
      queue_.FreeTensor(local);
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
    // Each segment starts at the aligned UB buffer base. Appending at an
    // arbitrary history-tail UB offset would violate DataCopyPad alignment.
    auto local = queue_.AllocTensor<uint16_t>();
    const AscendC::DataCopyExtParams copy{
        1, count * static_cast<uint32_t>(sizeof(uint16_t)), 0, 0, 0};
    const AscendC::DataCopyPadExtParams<uint16_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(local, source[source_offset], copy, pad);
    queue_.EnQue(local);
    local = queue_.DeQue<uint16_t>();
    AscendC::DataCopyPad(destination[destination_offset], local, copy);
    queue_.FreeTensor(local);
  }

  uint32_t cacheable_key_elements_ = 0;
  uint32_t cacheable_key_tasks_ = 0;
  T5SplitQkvAppendCacheTilingData tiling_{};
  AscendC::TPipe pipe_;
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2>
      queue_;
  AscendC::GlobalTensor<uint16_t> qkv_, past_key_, past_value_, query_, key_,
      key_bypass_, value_;
};
}  // namespace t5qkv
