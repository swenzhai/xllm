/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/t5_split_qkv_append_cache/common/t5_split_qkv_append_cache_tiling.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

namespace xllm::kernel::npu {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
t5_split_qkv_append_cache(const torch::Tensor& qkv,
                          const torch::Tensor& past_key,
                          const torch::Tensor& past_value,
                          int64_t head_num,
                          int64_t head_dim) {
  CHECK_EQ(qkv.dim(), 3);
  CHECK_EQ(past_key.dim(), 4);
  CHECK_EQ(past_value.sizes(), past_key.sizes());
  CHECK_EQ(qkv.device().type(), c10::DeviceType::PrivateUse1);
  CHECK_EQ(past_key.device(), qkv.device());
  CHECK_EQ(past_value.device(), qkv.device());
  CHECK(qkv.scalar_type() == torch::kFloat16 ||
        qkv.scalar_type() == torch::kBFloat16);
  CHECK_EQ(past_key.scalar_type(), qkv.scalar_type());
  CHECK_EQ(past_value.scalar_type(), qkv.scalar_type());
  CHECK(qkv.is_contiguous() && past_key.is_contiguous() &&
        past_value.is_contiguous());
  CHECK(at_npu::native::get_npu_format(qkv) == ACL_FORMAT_ND ||
        at_npu::native::get_npu_format(qkv) == ACL_FORMAT_NCHW);
  CHECK(at_npu::native::get_npu_format(past_key) == ACL_FORMAT_ND ||
        at_npu::native::get_npu_format(past_key) == ACL_FORMAT_NCHW);
  CHECK(at_npu::native::get_npu_format(past_value) == ACL_FORMAT_ND ||
        at_npu::native::get_npu_format(past_value) == ACL_FORMAT_NCHW);
  CHECK_EQ(past_key.size(1), head_num);
  CHECK_EQ(past_key.size(3), head_dim);
  T5SplitQkvAppendCacheTilingData tiling{};
  CHECK(t5qkv::make_tiling(past_key.size(0),
                           head_num,
                           past_key.size(2),
                           head_dim,
                           t5qkv::kMaxVectorCores,
                           tiling));
  CHECK_EQ(qkv.size(0), past_key.size(0));
  CHECK_EQ(qkv.size(1), 1);
  CHECK_EQ(qkv.size(2), 3 * head_num * head_dim);
  const c10_npu::NPUGuard guard(qkv.device());
  const auto stream = c10_npu::getCurrentNPUStream(qkv.device().index());
  torch::Tensor query =
      torch::empty({tiling.batch, head_num, 1, head_dim}, qkv.options());
  torch::Tensor key =
      torch::empty({tiling.batch, head_num, tiling.past_length + 1, head_dim},
                   qkv.options());
  torch::Tensor value = torch::empty_like(key);
  c10_npu::NPUCachingAllocator::recordStream(qkv.storage().data_ptr(), stream);
  c10_npu::NPUCachingAllocator::recordStream(past_key.storage().data_ptr(),
                                             stream);
  c10_npu::NPUCachingAllocator::recordStream(past_value.storage().data_ptr(),
                                             stream);
  c10_npu::NPUCachingAllocator::recordStream(query.storage().data_ptr(),
                                             stream);
  c10_npu::NPUCachingAllocator::recordStream(key.storage().data_ptr(), stream);
  c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(),
                                             stream);
  EXEC_NPU_CMD(aclnnT5SplitQkvAppendCache,
               qkv,
               past_key,
               past_value,
               head_num,
               head_dim,
               query,
               key,
               value);
  return {query, key, value};
}
}  // namespace xllm::kernel::npu
