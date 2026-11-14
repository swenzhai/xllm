# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.

set(ops_srcs ${CMAKE_CURRENT_SOURCE_DIR}/t5_split_qkv_append_cache.cpp)
npu_op_code_gen(SRC ${ops_srcs} PACKAGE ${package_name}
  OUT_DIR ${ASCEND_AUTOGEN_PATH} JOIN_OP_DEF True
  COMPILE_OPTIONS "-I${PROJECT_SOURCE_DIR}" -std=c++17)

file(GLOB autogen_aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN ${autogen_aclnn_src})
target_compile_options(cust_opapi PRIVATE -fvisibility=hidden)

file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_op_proto/*.cc)
file(GLOB proto_src ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
set_source_files_properties(${group_proto_src} ${proto_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH ${ops_srcs} ${group_proto_src} ${proto_src})
target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)

file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
set_source_files_properties(${fallback_src} PROPERTIES GENERATED TRUE)
npu_op_library(cust_optiling TILING ${ops_srcs} ${fallback_src})
target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)
npu_op_package_add(${package_name} LIBRARY cust_optiling cust_opapi cust_op_proto)
