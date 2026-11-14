# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.

cmake_minimum_required(VERSION 3.20)
project(t5_split_qkv_append_cache LANGUAGES CXX)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "Source the CANN set_env.sh before configuring this project")
endif()
list(APPEND CMAKE_PREFIX_PATH "$ENV{ASCEND_HOME_PATH}/lib64/cmake")
set(ASCEND_COMPUTE_UNIT "ascend910_93" CACHE STRING "Target Atlas A3 compute unit")
set(ENABLE_SOURCE_PACKAGE ON CACHE BOOL "Package Ascend C source")
set(ENABLE_BINARY_PACKAGE ON CACHE BOOL "Package compiled kernels")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN root")
set(ASCEND_PYTHON_EXECUTABLE "python3" CACHE STRING "Build Python interpreter")
set(vendor_name "t5qkv" CACHE STRING "Private operator vendor")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ASC REQUIRED)
set(package_name ${vendor_name})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

npu_op_package(${package_name} TYPE RUN CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR}/)
add_subdirectory(op_host)
add_subdirectory(op_kernel)
