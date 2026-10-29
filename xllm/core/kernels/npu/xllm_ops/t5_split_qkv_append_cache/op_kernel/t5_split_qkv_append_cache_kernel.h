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
    value_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(value),
                           query_elements * (tiling.past_length + 1));
    pipe_.InitBuffer(
        queue_, /*num=*/2, tiling.tile_elements * sizeof(uint16_t));
  }

  __aicore__ inline void process() {
    for (uint32_t task_id = AscendC::GetBlockIdx();
         task_id < tiling_.total_tasks;
         task_id += tiling_.block_dim) {
      const CopyTask task = decode_task(tiling_, task_id);
      if (task.output == Output::QUERY) {
        copy_segment(query_,
                     qkv_,
                     task.destination_offset,
                     task.source_offset,
                     task.count);
      } else if (task.output == Output::KEY) {
        copy_segment(key_,
                     past_key_,
                     task.destination_offset,
                     task.source_offset,
                     task.count);
        copy_segment(key_,
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
  }

 private:
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

  T5SplitQkvAppendCacheTilingData tiling_{};
  AscendC::TPipe pipe_;
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2>
      queue_;
  AscendC::GlobalTensor<uint16_t> qkv_, past_key_, past_value_, query_, key_,
      value_;
};
}  // namespace t5qkv
