#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${T5QKV_BUILD_DIR:-${project_dir}/build}"
: "${ASCEND_HOME_PATH:?Source the CANN set_env.sh before building}"
cmake -S "${project_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_COMPUTE_UNIT=ascend910_93 \
  -Dvendor_name=t5qkv \
  -DASCEND_PYTHON_EXECUTABLE="$(command -v python3)"
cmake --build "${build_dir}" --target binary package --parallel "${T5QKV_BUILD_JOBS:-8}"
