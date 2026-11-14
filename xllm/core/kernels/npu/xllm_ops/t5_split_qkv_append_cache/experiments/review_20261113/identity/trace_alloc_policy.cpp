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
// Diagnostic interposition only. Never preload in acceptance or production.
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "acl/acl_rt.h"

namespace {
using SymbolLookup = void* (*)(void*, const char*);
std::atomic<void*> malloc_real{nullptr};
std::atomic<void*> aligned_real{nullptr};
std::atomic<void*> physical_real{nullptr};

void* lookup_real(void* handle, const char* name) {
  // aarch64 glibc's initial symbol version. Missing support is a hard
  // diagnostic failure, not a fallback or substitute allocation.
  const auto lookup =
      reinterpret_cast<SymbolLookup>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17"));
  if (lookup == nullptr) {
    std::abort();
  }
  return lookup(handle, name);
}

template <typename Function>
Function real_function(std::atomic<void*>& slot, const char* name) {
  void* symbol = slot.load(std::memory_order_acquire);
  if (symbol == nullptr) {
    symbol = lookup_real(RTLD_NEXT, name);
  }
  if (symbol == nullptr) {
    std::abort();
  }
  return reinterpret_cast<Function>(symbol);
}

void record(const char* function,
            size_t size,
            int32_t policy,
            aclError rc,
            void* address,
            int32_t handle_type = -1,
            int32_t allocation_type = -1,
            int32_t attribute = -1) {
  const char* path = std::getenv("T5QKV_ALLOC_TRACE");
  if (path == nullptr) {
    std::abort();
  }
  char data[512];
  const int32_t length = std::snprintf(
      data,
      sizeof(data),
      "{\"function\":\"%s\",\"bytes\":%zu,\"policy_raw\":%d,\"rc\":%d,"
      "\"pointer_or_handle\":\"%p\",\"handle_type\":%d,"
      "\"allocation_type\":%d,\"mem_attr\":%d}\n",
      function,
      size,
      policy,
      rc,
      address,
      handle_type,
      allocation_type,
      attribute);
  const int32_t fd =
      open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (length <= 0 || static_cast<size_t>(length) >= sizeof(data) || fd < 0) {
    std::abort();
  }
  const ssize_t written = write(fd, data, static_cast<size_t>(length));
  close(fd);
  if (written != length) {
    std::abort();
  }
}
}  // namespace

extern "C" aclError aclrtMalloc(void** pointer,
                                size_t size,
                                aclrtMemMallocPolicy policy) {
  const aclError rc = real_function<decltype(&aclrtMalloc)>(
      malloc_real, "aclrtMalloc")(pointer, size, policy);
  record("aclrtMalloc",
         size,
         static_cast<int32_t>(policy),
         rc,
         rc == 0 ? *pointer : nullptr);
  return rc;
}

extern "C" aclError aclrtMallocAlign32(void** pointer,
                                       size_t size,
                                       aclrtMemMallocPolicy policy) {
  const aclError rc = real_function<decltype(&aclrtMallocAlign32)>(
      aligned_real, "aclrtMallocAlign32")(pointer, size, policy);
  record("aclrtMallocAlign32",
         size,
         static_cast<int32_t>(policy),
         rc,
         rc == 0 ? *pointer : nullptr);
  return rc;
}

extern "C" aclError aclrtMallocPhysical(aclrtDrvMemHandle* handle,
                                        size_t size,
                                        const aclrtPhysicalMemProp* property,
                                        uint64_t flags) {
  const aclError rc = real_function<decltype(&aclrtMallocPhysical)>(
      physical_real, "aclrtMallocPhysical")(handle, size, property, flags);
  record("aclrtMallocPhysical",
         size,
         -1,
         rc,
         rc == 0 ? *handle : nullptr,
         static_cast<int32_t>(property->handleType),
         static_cast<int32_t>(property->allocationType),
         static_cast<int32_t>(property->memAttr));
  return rc;
}

extern "C" void* dlsym(void* handle, const char* name) noexcept {
  void* symbol = lookup_real(handle, name);
  if (symbol == nullptr) {
    return nullptr;
  }
  if (std::strcmp(name, "aclrtMalloc") == 0) {
    if (symbol == reinterpret_cast<void*>(&aclrtMalloc)) {
      symbol = lookup_real(RTLD_NEXT, name);
    }
    if (symbol == nullptr) {
      return nullptr;
    }
    malloc_real.store(symbol, std::memory_order_release);
    return reinterpret_cast<void*>(&aclrtMalloc);
  }
  if (std::strcmp(name, "aclrtMallocAlign32") == 0) {
    if (symbol == reinterpret_cast<void*>(&aclrtMallocAlign32)) {
      symbol = lookup_real(RTLD_NEXT, name);
    }
    if (symbol == nullptr) {
      return nullptr;
    }
    aligned_real.store(symbol, std::memory_order_release);
    return reinterpret_cast<void*>(&aclrtMallocAlign32);
  }
  if (std::strcmp(name, "aclrtMallocPhysical") == 0) {
    if (symbol == reinterpret_cast<void*>(&aclrtMallocPhysical)) {
      symbol = lookup_real(RTLD_NEXT, name);
    }
    if (symbol == nullptr) {
      return nullptr;
    }
    physical_real.store(symbol, std::memory_order_release);
    return reinterpret_cast<void*>(&aclrtMallocPhysical);
  }
  return symbol;
}
