#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?Load the CANN environment first}"
export CC=${CC:-gcc}
export CXX=${CXX:-g++}
export MAX_JOBS=${MAX_JOBS:-2}
# CANN's legacy preprocessing transforms object files in place; use a fresh
# build tree for each direct-kernel build instead of reprocessing old objects.
direct_build=$(mktemp -d "$project_root/build_direct.XXXXXX")
cmake -S "$project_root/direct" -B "$direct_build" -DSOC_VERSION=Ascend910_9382
cmake --build "$direct_build" --parallel "$MAX_JOBS"
mkdir -p "$project_root/build_direct"
cp "$direct_build/libt5qkv_direct.so" "$project_root/build_direct/libt5qkv_direct.so.new"
mv "$project_root/build_direct/libt5qkv_direct.so.new" "$project_root/build_direct/libt5qkv_direct.so"
cd "$project_root"
# Enter the compiler dependency check even when the extension itself is newer
# than binding.cpp: changes to the shared tiling header must not be skipped.
python3 framework/setup.py build_ext --build-temp build_framework/temp --build-lib build_framework/lib --inplace --force
