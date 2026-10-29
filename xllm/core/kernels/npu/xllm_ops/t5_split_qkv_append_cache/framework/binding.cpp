/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or
agreed to in writing, software is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND.
==============================================================================*/

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/t5_split_qkv_append_cache_tiling.h"

struct aclOpExecutor;
extern "C" uint32_t t5qkv_launch(uint32_t block_dim,
                                 void* stream,
                                 void* qkv,
                                 void* past_key,
                                 void* past_value,
                                 void* query,
                                 void* key,
                                 void* value,
                                 T5SplitQkvAppendCacheTilingData tiling);

namespace {
namespace py = pybind11;
using Tensors = std::vector<torch::Tensor>;
using QueryFn = int32_t (*)(const aclTensor*,
                            const aclTensor*,
                            const aclTensor*,
                            int64_t,
                            int64_t,
                            const aclTensor*,
                            const aclTensor*,
                            const aclTensor*,
                            uint64_t*,
                            aclOpExecutor**);
using ExecuteFn = int32_t (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);

// Exceptions here translate invalid caller metadata and C API failures at the
// Python boundary; no failing operation is caught or retried as another path.
void require(bool condition, const char* message) {
  if (!condition) {
    throw py::value_error(message);
  }
}

void check_acl(int32_t status, const char* operation) {
  if (status != 0) {
    const char* detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(operation) + " failed (" +
                             std::to_string(status) +
                             "): " + (detail == nullptr ? "" : detail));
  }
}

class OperatorApi final {
 public:
  OperatorApi() {
    const char* library = std::getenv("T5QKV_OPAPI_LIB");
    require(library != nullptr && library[0] != '\0',
            "Set T5QKV_OPAPI_LIB to the private op_api/lib/libcust_opapi.so");
    handle_ = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      throw std::runtime_error(dlerror());
    }
    query = reinterpret_cast<QueryFn>(
        dlsym(handle_, "aclnnT5SplitQkvAppendCacheGetWorkspaceSize"));
    execute = reinterpret_cast<ExecuteFn>(
        dlsym(handle_, "aclnnT5SplitQkvAppendCache"));
    if (query == nullptr || execute == nullptr) {
      dlclose(handle_);
      handle_ = nullptr;
      throw py::value_error("T5QKV ACLNN symbols missing");
    }
  }

  // Keep the registered operator library loaded until process exit, since
  // asynchronous launches and CANN registries can outlive individual calls.
  QueryFn query = nullptr;
  ExecuteFn execute = nullptr;

 private:
  void* handle_ = nullptr;
};

OperatorApi& api() {
  static OperatorApi instance;
  return instance;
}

// torch_npu labels ordinary dense rank-four tensors NCHW; their contiguous
// bytes have the same layout as our ND descriptors. Packed formats are
// rejected.
bool is_dense_format(const torch::Tensor& tensor) {
  const int64_t format = at_npu::native::get_npu_format(tensor);
  return format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW;
}

T5SplitQkvAppendCacheTilingData validate_inputs(const Tensors& inputs,
                                                int64_t head_num,
                                                int64_t head_dim) {
  require(inputs.size() == 3, "exactly three inputs are required");
  for (const auto& input : inputs) {
    require(input.defined(), "input is undefined");
    require(input.device().type() == c10::DeviceType::PrivateUse1,
            "inputs must be NPU tensors");
    require(input.device() == inputs[0].device(), "input devices differ");
    require(input.scalar_type() == torch::kFloat16 ||
                input.scalar_type() == torch::kBFloat16,
            "only FP16 and BF16 are supported");
    require(input.scalar_type() == inputs[0].scalar_type(),
            "input dtypes differ");
    require(input.is_contiguous(), "inputs must be contiguous");
    require(is_dense_format(input), "inputs must use ND storage");
  }
  const auto& qkv = inputs[0];
  const auto& past_key = inputs[1];
  const auto& past_value = inputs[2];
  require(qkv.dim() == 3 && past_key.dim() == 4 && past_value.dim() == 4,
          "input ranks must be 3, 4, 4");
  require(past_key.sizes() == past_value.sizes(), "cache shapes differ");
  require(past_key.size(1) == head_num && past_key.size(3) == head_dim,
          "attributes disagree with cache shape");
  T5SplitQkvAppendCacheTilingData tiling{};
  require(t5qkv::make_tiling(past_key.size(0),
                             head_num,
                             past_key.size(2),
                             head_dim,
                             t5qkv::kMaxVectorCores,
                             tiling),
          "shape exceeds supported limits");
  require(qkv.size(0) == past_key.size(0) && qkv.size(1) == 1 &&
              qkv.size(2) == 3 * head_num * head_dim,
          "packed QKV shape mismatch");
  return tiling;
}

void validate_outputs(const Tensors& inputs,
                      const Tensors& outputs,
                      const T5SplitQkvAppendCacheTilingData& tiling) {
  require(outputs.size() == 3, "exactly three outputs are required");
  std::array<const void*, 3> storage{};
  for (size_t index = 0; index < outputs.size(); ++index) {
    const auto& output = outputs[index];
    require(output.defined() && output.device() == inputs[0].device(),
            "output must be on the input device");
    require(output.scalar_type() == inputs[0].scalar_type(),
            "output dtype mismatch");
    require(output.is_contiguous() && is_dense_format(output),
            "output must be contiguous ND");
    require(output.dim() == 4 && output.size(0) == tiling.batch &&
                output.size(1) == tiling.head_num &&
                output.size(3) == tiling.head_dim &&
                output.size(2) == (index == 0 ? 1 : tiling.past_length + 1),
            "output shape mismatch");
    storage[index] = output.storage().data_ptr().get();
    for (const auto& input : inputs) {
      require(input.numel() == 0 ||
                  input.storage().data_ptr().get() != storage[index],
              "output shares input storage");
    }
    for (size_t previous = 0; previous < index; ++previous) {
      require(storage[index] != storage[previous], "outputs share storage");
    }
  }
}

class TensorDeleter final {
 public:
  void operator()(aclTensor* tensor) const {
    if (tensor != nullptr) {
      aclDestroyTensor(tensor);
    }
  }
};
using TensorHandle = std::unique_ptr<aclTensor, TensorDeleter>;

TensorHandle describe(const torch::Tensor& tensor) {
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  const aclDataType dtype =
      tensor.scalar_type() == torch::kFloat16 ? ACL_FLOAT16 : ACL_BF16;
  // data_ptr already includes storage_offset; offset zero describes this exact
  // contiguous view without exposing or copying any prefix/suffix allocation.
  TensorHandle result(aclCreateTensor(sizes.data(),
                                      sizes.size(),
                                      dtype,
                                      strides.data(),
                                      /*offset=*/0,
                                      ACL_FORMAT_ND,
                                      sizes.data(),
                                      sizes.size(),
                                      tensor.data_ptr()));
  require(result != nullptr, "aclCreateTensor failed");
  return result;
}

std::vector<int64_t> validate(const Tensors& inputs,
                              int64_t head_num,
                              int64_t head_dim) {
  const auto tiling = validate_inputs(inputs, head_num, head_dim);
  return {tiling.batch,
          tiling.head_num,
          tiling.past_length,
          tiling.head_dim,
          inputs[0].device().index(),
          static_cast<int64_t>(inputs[0].scalar_type())};
}

void validate_out(const Tensors& inputs,
                  const Tensors& outputs,
                  int64_t head_num,
                  int64_t head_dim) {
  validate_outputs(
      inputs, outputs, validate_inputs(inputs, head_num, head_dim));
}

Tensors run(const Tensors& inputs,
            int64_t head_num,
            int64_t head_dim,
            bool direct) {
  const auto tiling = validate_inputs(inputs, head_num, head_dim);
  const c10_npu::NPUGuard device_guard(inputs[0].device());
  const auto stream = c10_npu::getCurrentNPUStream(inputs[0].device().index());
  // Drain pending framework submissions before bypassing its host task queue.
  // This does not synchronize device work.
  const aclrtStream acl_stream = stream.stream();
  Tensors outputs;
  outputs.reserve(3);
  outputs.emplace_back(
      torch::empty({tiling.batch, tiling.head_num, 1, tiling.head_dim},
                   inputs[0].options()));
  outputs.emplace_back(torch::empty(
      {tiling.batch, tiling.head_num, tiling.past_length + 1, tiling.head_dim},
      inputs[0].options()));
  outputs.emplace_back(torch::empty_like(outputs[1]));
  for (const auto& input : inputs) {
    c10_npu::NPUCachingAllocator::recordStream(input.storage().data_ptr(),
                                               stream);
  }
  for (const auto& output : outputs) {
    c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(),
                                               stream);
  }
  if (direct) {
    check_acl(t5qkv_launch(tiling.block_dim,
                           acl_stream,
                           inputs[0].data_ptr(),
                           inputs[1].data_ptr(),
                           inputs[2].data_ptr(),
                           outputs[0].data_ptr(),
                           outputs[1].data_ptr(),
                           outputs[2].data_ptr(),
                           tiling),
              "t5qkv_launch");
  } else {
    std::array<TensorHandle, 6> descriptors;
    for (size_t index = 0; index < 3; ++index) {
      descriptors[index] = describe(inputs[index]);
      descriptors[index + 3] = describe(outputs[index]);
    }
    uint64_t workspace = 0;
    aclOpExecutor* executor = nullptr;
    check_acl(api().query(descriptors[0].get(),
                          descriptors[1].get(),
                          descriptors[2].get(),
                          head_num,
                          head_dim,
                          descriptors[3].get(),
                          descriptors[4].get(),
                          descriptors[5].get(),
                          &workspace,
                          &executor),
              "aclnnT5SplitQkvAppendCacheGetWorkspaceSize");
    require(workspace == 0, "operator requested nonzero global workspace");
    check_acl(api().execute(nullptr, workspace, executor, acl_stream),
              "aclnnT5SplitQkvAppendCache");
  }
  return outputs;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &run);
  module.def("validate", &validate);
  module.def("validate_out", &validate_out);
}
