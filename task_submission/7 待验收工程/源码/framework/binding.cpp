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
#include <torch_npu/csrc/framework/OpCommand.h>

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

struct TensorMetadata {
  std::array<int64_t, 4> sizes{};
  std::array<int64_t, 4> strides{};
  uint64_t rank = 0;
  aclDataType dtype = ACL_FLOAT16;
  void* data = nullptr;
};

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

TensorMetadata tensor_metadata(const torch::Tensor& tensor) {
  TensorMetadata metadata{};
  metadata.rank = tensor.dim();
  for (uint64_t index = 0; index < metadata.rank; ++index) {
    metadata.sizes[index] = tensor.size(index);
    metadata.strides[index] = tensor.stride(index);
  }
  metadata.dtype =
      tensor.scalar_type() == torch::kFloat16 ? ACL_FLOAT16 : ACL_BF16;
  metadata.data = tensor.data_ptr();
  return metadata;
}

TensorHandle describe(const TensorMetadata& tensor) {
  // data_ptr already includes storage_offset; offset zero describes this exact
  // contiguous view without exposing or copying any prefix/suffix allocation.
  TensorHandle result(aclCreateTensor(tensor.sizes.data(),
                                      tensor.rank,
                                      tensor.dtype,
                                      tensor.strides.data(),
                                      /*offset=*/0,
                                      ACL_FORMAT_ND,
                                      tensor.sizes.data(),
                                      tensor.rank,
                                      tensor.data));
  if (result == nullptr) {
    throw std::runtime_error("aclCreateTensor failed");
  }
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
  // Use the framework task queue, preserving ordering without draining that
  // queue on every call. The allocator tracks input use until completion.
  const aclrtStream acl_stream = stream.stream(false);
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
  // Fresh outputs belong to this allocation stream. The caching allocator
  // reuses them only on the same stream, after this queued launch; registering
  // that origin stream again would add redundant completion events per output.
  OperatorApi* operator_api = direct ? nullptr : &api();
  std::array<TensorMetadata, 6> metadata;
  for (size_t index = 0; index < 3; ++index) {
    metadata[index] = tensor_metadata(inputs[index]);
    metadata[index + 3] = tensor_metadata(outputs[index]);
  }
  auto launch =
      [metadata, tiling, head_num, head_dim, direct, acl_stream, operator_api]()
      -> int {
    // This task is queued before the caller can release its tensor references.
    // recordStream makes subsequent frees enqueue allocator events behind this
    // task. No Python-associated owner is retained or destroyed on the worker.
    if (direct) {
      check_acl(t5qkv_launch(tiling.block_dim,
                             acl_stream,
                             metadata[0].data,
                             metadata[1].data,
                             metadata[2].data,
                             metadata[3].data,
                             metadata[4].data,
                             metadata[5].data,
                             tiling),
                "t5qkv_launch");
    } else {
      std::array<TensorHandle, 6> descriptors;
      for (size_t index = 0; index < descriptors.size(); ++index) {
        descriptors[index] = describe(metadata[index]);
      }
      uint64_t workspace = 0;
      aclOpExecutor* executor = nullptr;
      check_acl(operator_api->query(descriptors[0].get(),
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
      if (workspace != 0) {
        throw std::runtime_error("operator requested nonzero global workspace");
      }
      check_acl(operator_api->execute(nullptr, workspace, executor, acl_stream),
                "aclnnT5SplitQkvAppendCache");
    }
    return 0;
  };
  at_npu::native::OpCommand::RunOpApi("T5SplitQkvAppendCache", launch);
  return outputs;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &run);
  module.def("validate", &validate);
  module.def("validate_out", &validate_out);
}
