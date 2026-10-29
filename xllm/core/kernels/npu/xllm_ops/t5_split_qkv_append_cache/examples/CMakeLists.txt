# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
cmake_minimum_required(VERSION 3.20)
project(t5qkv_aclnn_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(T5QKV_OPP_PATH "" CACHE PATH "Installed vendor directory containing op_api")
if(NOT T5QKV_OPP_PATH OR NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "Set T5QKV_OPP_PATH and source the CANN environment")
endif()
add_executable(t5qkv_aclnn_example aclnn_example.cpp)
target_include_directories(t5qkv_aclnn_example PRIVATE
  "$ENV{ASCEND_HOME_PATH}/include" "${T5QKV_OPP_PATH}/op_api/include")
target_link_directories(t5qkv_aclnn_example PRIVATE
  "${T5QKV_OPP_PATH}/op_api/lib" "$ENV{ASCEND_HOME_PATH}/lib64")
target_link_libraries(t5qkv_aclnn_example PRIVATE cust_opapi ascendcl nnopbase)
set_target_properties(t5qkv_aclnn_example PROPERTIES
  BUILD_RPATH "${T5QKV_OPP_PATH}/op_api/lib;$ENV{ASCEND_HOME_PATH}/lib64")
