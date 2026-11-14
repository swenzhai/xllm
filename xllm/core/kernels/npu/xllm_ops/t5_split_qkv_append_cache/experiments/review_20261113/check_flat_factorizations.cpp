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
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "common/t5_split_qkv_append_cache_tiling.h"

int32_t main() {
  uint32_t shapes = 0;
  uint32_t rejected_shapes = 0;
  uint32_t append_checks = 0;
  uint32_t query_checks = 0;
  for (uint32_t heads = 1; heads <= 128; heads *= 2) {
    const uint32_t batch = 256 / heads;
    T5SplitQkvAppendCacheTilingData tiling{};
    if (batch > 64) {
      assert(!t5qkv::make_tiling(batch, heads, 2048, 64, 48, tiling));
      ++rejected_shapes;
      continue;
    }
    assert(t5qkv::make_tiling(batch, heads, 2048, 64, 48, tiling));
    assert(t5qkv::uses_flat_history(tiling));
    for (uint32_t row = 0; row < 256; ++row) {
      for (bool value : {false, true}) {
        const auto piece = t5qkv::flat_history_piece(
            tiling, row * 2049 * 64 + 2048 * 64, 64, value);
        const uint32_t expected = row / heads * 3 * heads * 64 +
                                  (value ? 2 : 1) * heads * 64 +
                                  row % heads * 64;
        assert(piece.qkv && piece.count == 64 &&
               piece.source_offset == expected);
        ++append_checks;
      }
    }
    std::vector<uint32_t> seen(batch, 0);
    for (uint32_t core = 0; core < 48; ++core) {
      const auto range = t5qkv::interleaved_query_range(tiling, core);
      for (uint32_t q = range.begin; q < range.end; ++q) {
        assert(q < seen.size());
        ++seen[q];
        const auto task =
            t5qkv::decode_task(tiling, 2 * tiling.cache_tasks_per_tensor + q);
        assert(task.output == t5qkv::Output::QUERY &&
               task.source_offset == q * 3 * heads * 64 &&
               task.destination_offset == q * heads * 64 &&
               task.count == heads * 64);
        ++query_checks;
      }
    }
    for (uint32_t n : seen) {
      assert(n == 1);
    }
    ++shapes;
  }
  assert(shapes == 6 && rejected_shapes == 2);
  std::cout << "{\"scope\":\"CPU_flat_guard_all_batch_head_factorizations\","
               "\"npu_executed\":false,\"status\":\"passed\",\"shapes\":"
            << shapes << ",\"rejected_shapes\":" << rejected_shapes
            << ",\"append_address_checks\":" << append_checks
            << ",\"query_task_checks\":" << query_checks << "}\n";
}
