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

"""Compile and execute the shared C++ mapper without torch or CANN.

This checks address coverage and boundaries only. It does not emulate DMA or
establish device correctness/performance. Compiler and checker output is passed
through unchanged; the checker emits a machine-readable JSON result.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++", help="Host C++17 compiler")
    args = parser.parse_args()
    compiler = shutil.which(args.cxx)
    if compiler is None:
        parser.error(f"C++ compiler not found: {args.cxx}")
    project_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="t5qkv-mapping-") as temporary:
        executable = Path(temporary) / "check_task_mapping"
        subprocess.run(
            [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O2",
             "-I", str(project_root),
             str(project_root / "tests" / "check_task_mapping.cpp"),
             "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    _main()
