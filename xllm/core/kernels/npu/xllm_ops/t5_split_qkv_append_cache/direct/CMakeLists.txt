# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cmake_minimum_required(VERSION 3.18)
project(t5qkv_direct LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SOC_VERSION "Ascend910_9382" CACHE STRING "Target Ascend SoC")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit root")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
        CACHE PATH "CANN toolkit root")
  endif()
endif()
set(ASCENDC_CMAKE "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
if(NOT EXISTS "${ASCENDC_CMAKE}")
  message(FATAL_ERROR "Missing Ascend C CMake support: ${ASCENDC_CMAKE}")
endif()
include("${ASCENDC_CMAKE}")

get_filename_component(T5QKV_PROJECT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
ascendc_library(t5qkv_direct SHARED
  "${T5QKV_PROJECT_ROOT}/op_kernel/t5_split_qkv_append_cache.cpp")
ascendc_compile_definitions(t5qkv_direct PRIVATE T5QKV_DIRECT_LAUNCH=1)
ascendc_include_directories(t5qkv_direct PRIVATE "${T5QKV_PROJECT_ROOT}")
set_target_properties(t5qkv_direct PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Compile the public C ABI with the host compiler. Its generated launcher accepts
# a host pointer to the by-value kernel tiling payload.
target_sources(t5qkv_direct PRIVATE "${CMAKE_CURRENT_LIST_DIR}/launch.cpp")
target_include_directories(t5qkv_direct PRIVATE "${T5QKV_PROJECT_ROOT}")
