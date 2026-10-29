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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "aclnn_t5_split_qkv_append_cache.h"

namespace {

bool check(aclError status, const char* operation) {
  if (status == ACL_SUCCESS) {
    return true;
  }
  std::cerr << operation << " failed: " << status << '\n';
  return false;
}

class Runtime final {
 public:
  Runtime() = default;
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  ~Runtime() {
    if (stream_ != nullptr) {
      aclrtDestroyStream(stream_);
    }
    if (device_selected_) {
      aclrtResetDevice(device_);
    }
    if (initialized_) {
      aclFinalize();
    }
  }

  bool initialize(int32_t device) {
    device_ = device;
    initialized_ = check(aclInit(nullptr), "aclInit");
    if (!initialized_) {
      return false;
    }
    device_selected_ = check(aclrtSetDevice(device_), "aclrtSetDevice");
    return device_selected_ &&
           check(aclrtCreateStream(&stream_), "aclrtCreateStream");
  }
  aclrtStream stream() const { return stream_; }

 private:
  int32_t device_ = 0;
  bool initialized_ = false;
  bool device_selected_ = false;
  aclrtStream stream_ = nullptr;
};

class DeviceDeleter final {
 public:
  void operator()(void* pointer) const { aclrtFree(pointer); }
};

class TensorDeleter final {
 public:
  void operator()(aclTensor* tensor) const { aclDestroyTensor(tensor); }
};

class DeviceTensor final {
 public:
  DeviceTensor() = default;
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  bool initialize(std::vector<int64_t> shape,
                  aclDataType dtype,
                  const std::vector<uint16_t>* bits = nullptr) {
    shape_ = std::move(shape);
    count_ = std::accumulate(
        shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<int64_t>());
    void* pointer = nullptr;
    if (!check(aclrtMalloc(&pointer,
                           std::max<int64_t>(count_, 1) * sizeof(uint16_t),
                           ACL_MEM_MALLOC_NORMAL_ONLY),
               "aclrtMalloc")) {
      return false;
    }
    data_.reset(pointer);
    std::vector<int64_t> strides(shape_.size(), 1);
    for (size_t index = shape_.size() - 1; index > 0; --index) {
      strides[index - 1] = strides[index] * shape_[index];
    }
    tensor_.reset(aclCreateTensor(shape_.data(),
                                  shape_.size(),
                                  dtype,
                                  strides.data(),
                                  /*offset=*/0,
                                  ACL_FORMAT_ND,
                                  shape_.data(),
                                  shape_.size(),
                                  data_.get()));
    if (tensor_ == nullptr) {
      std::cerr << "aclCreateTensor failed\n";
      return false;
    }
    return bits == nullptr || (bits->size() == static_cast<size_t>(count_) &&
                               check(aclrtMemcpy(data_.get(),
                                                 count_ * sizeof(uint16_t),
                                                 bits->data(),
                                                 count_ * sizeof(uint16_t),
                                                 ACL_MEMCPY_HOST_TO_DEVICE),
                                     "aclrtMemcpy H2D"));
  }

  aclTensor* tensor() const { return tensor_.get(); }

  bool equals(const std::vector<uint16_t>& expected) const {
    std::vector<uint16_t> actual(static_cast<size_t>(count_));
    return check(aclrtMemcpy(actual.data(),
                             count_ * sizeof(uint16_t),
                             data_.get(),
                             count_ * sizeof(uint16_t),
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy D2H") &&
           actual == expected;
  }

 private:
  std::vector<int64_t> shape_;
  int64_t count_ = 0;
  std::unique_ptr<void, DeviceDeleter> data_;
  std::unique_ptr<aclTensor, TensorDeleter> tensor_;
};

bool run_example(aclrtStream stream, aclDataType dtype) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kHeads = 3;
  constexpr int64_t kLength = 2;
  constexpr int64_t kDim = 5;
  constexpr int64_t kHeadElements = kHeads * kDim;
  std::vector<uint16_t> qkv(kBatch * 3 * kHeadElements);
  std::vector<uint16_t> past_key(kBatch * kHeads * kLength * kDim);
  std::vector<uint16_t> past_value(past_key.size());
  for (size_t index = 0; index < qkv.size(); ++index) {
    qkv[index] = static_cast<uint16_t>(0x7c01U ^ (index * 97U));
  }
  for (size_t index = 0; index < past_key.size(); ++index) {
    past_key[index] = static_cast<uint16_t>(0x8000U ^ (index * 53U));
    past_value[index] = static_cast<uint16_t>(0x7f81U ^ (index * 31U));
  }
  std::vector<uint16_t> expected_query(kBatch * kHeadElements);
  std::vector<uint16_t> expected_key(kBatch * kHeads * (kLength + 1) * kDim);
  std::vector<uint16_t> expected_value(expected_key.size());
  for (int64_t batch = 0; batch < kBatch; ++batch) {
    std::copy_n(qkv.begin() + batch * 3 * kHeadElements,
                kHeadElements,
                expected_query.begin() + batch * kHeadElements);
    for (int64_t head = 0; head < kHeads; ++head) {
      const int64_t row = batch * kHeads + head;
      const int64_t output = row * (kLength + 1) * kDim;
      std::copy_n(past_key.begin() + row * kLength * kDim,
                  kLength * kDim,
                  expected_key.begin() + output);
      std::copy_n(past_value.begin() + row * kLength * kDim,
                  kLength * kDim,
                  expected_value.begin() + output);
      const int64_t packed = batch * 3 * kHeadElements + head * kDim;
      std::copy_n(qkv.begin() + packed + kHeadElements,
                  kDim,
                  expected_key.begin() + output + kLength * kDim);
      std::copy_n(qkv.begin() + packed + 2 * kHeadElements,
                  kDim,
                  expected_value.begin() + output + kLength * kDim);
    }
  }

  std::array<DeviceTensor, 6> tensors;
  if (!tensors[0].initialize({kBatch, 1, 3 * kHeadElements}, dtype, &qkv) ||
      !tensors[1].initialize(
          {kBatch, kHeads, kLength, kDim}, dtype, &past_key) ||
      !tensors[2].initialize(
          {kBatch, kHeads, kLength, kDim}, dtype, &past_value) ||
      !tensors[3].initialize({kBatch, kHeads, 1, kDim}, dtype) ||
      !tensors[4].initialize({kBatch, kHeads, kLength + 1, kDim}, dtype) ||
      !tensors[5].initialize({kBatch, kHeads, kLength + 1, kDim}, dtype)) {
    return false;
  }
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  if (!check(aclnnT5SplitQkvAppendCacheGetWorkspaceSize(tensors[0].tensor(),
                                                        tensors[1].tensor(),
                                                        tensors[2].tensor(),
                                                        kHeads,
                                                        kDim,
                                                        tensors[3].tensor(),
                                                        tensors[4].tensor(),
                                                        tensors[5].tensor(),
                                                        &workspace_size,
                                                        &executor),
             "aclnnT5SplitQkvAppendCacheGetWorkspaceSize")) {
    return false;
  }
  if (workspace_size != 0) {
    std::cerr << "Expected zero workspace, received " << workspace_size << '\n';
    return false;
  }
  if (!check(
          aclnnT5SplitQkvAppendCache(nullptr, workspace_size, executor, stream),
          "aclnnT5SplitQkvAppendCache") ||
      !check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
    return false;
  }
  const bool correct =
      tensors[0].equals(qkv) && tensors[1].equals(past_key) &&
      tensors[2].equals(past_value) && tensors[3].equals(expected_query) &&
      tensors[4].equals(expected_key) && tensors[5].equals(expected_value);
  std::cout << "dtype=" << static_cast<int32_t>(dtype)
            << " bit_exact=" << correct << " workspace_bytes=" << workspace_size
            << '\n';
  return correct;
}

}  // namespace

int main() {
  Runtime runtime;
  if (!runtime.initialize(/*device=*/0)) {
    return EXIT_FAILURE;
  }
  return run_example(runtime.stream(), ACL_FLOAT16) &&
                 run_example(runtime.stream(), ACL_BF16)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
