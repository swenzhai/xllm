# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Build from the operator project root: python framework/setup.py build_ext --inplace."""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension

ROOT = Path(__file__).resolve().parent.parent
CANN = Path(os.environ["ASCEND_HOME_PATH"])
DIRECT = Path(os.environ.get("T5QKV_DIRECT_DIR", str(ROOT / "build_direct")))

setup(
    name="xllm-t5qkv",
    ext_modules=[
        NpuExtension(
            "_t5qkv",
            sources=[str(ROOT / "framework/binding.cpp")],
            include_dirs=[str(ROOT), str(CANN / "include")],
            library_dirs=[str(DIRECT), str(CANN / "lib64")],
            libraries=["t5qkv_direct", "ascendcl", "nnopbase", "dl"],
            runtime_library_dirs=[str(DIRECT), str(CANN / "lib64")],
            extra_compile_args=["-O2", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
