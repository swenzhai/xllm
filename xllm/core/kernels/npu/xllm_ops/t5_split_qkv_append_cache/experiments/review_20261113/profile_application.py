# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Bootstrap the selected venv when msprof resolves its interpreter symlink.

Diagnostic only: run the unchanged public program, preserving its exit status.
"""

import argparse
import os
from pathlib import Path
import runpy
import site
import sys

from evidence import _write


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.identity.exists():
        parser.error("Choose a new application identity record")
    site.addsitedir(str(args.site_packages.resolve()))
    benchmark = args.benchmark.resolve()
    remainder = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(benchmark), *remainder]
    sys.path.insert(0, str(benchmark.parent))
    _write(args.identity, {"scope": "profiler_application_bootstrap_not_acceptance",
                          "python": sys.version, "executable": sys.executable,
                          "prefix": sys.prefix, "virtual_env": os.environ.get("VIRTUAL_ENV"),
                          "explicit_site_packages": str(args.site_packages.resolve()),
                          "argv": sys.argv, "sys_path": sys.path})
    runpy.run_path(str(benchmark), run_name="__main__")


if __name__ == "__main__":
    _main()
