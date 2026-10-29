# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0)
endif()
npu_op_kernel_options(ascendc_kernels ALL OPTIONS "-I${PROJECT_SOURCE_DIR}")
npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
npu_op_kernel_library(ascendc_kernels SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
  TILING_LIBRARY cust_optiling)
npu_op_package_add(${package_name} LIBRARY ascendc_kernels)

# The generated source entry is flattened into dynamic/. Preserve the quoted
# include hierarchy there so the installed sources do not need the build tree.
set(dynamic_package_path "op_impl/ai_core/tbe/${vendor_name}_impl/dynamic")
npu_op_package_add(${package_name}
  FILES ${PROJECT_SOURCE_DIR}/op_kernel/t5_split_qkv_append_cache_kernel.h
  PACKAGE_PATH "${dynamic_package_path}/op_kernel")
npu_op_package_add(${package_name}
  FILES ${PROJECT_SOURCE_DIR}/common/t5_split_qkv_append_cache_tiling.h
  PACKAGE_PATH "${dynamic_package_path}/op_kernel/common")
