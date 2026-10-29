#!/usr/bin/env bash
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /absolute/path/to/package.run /absolute/private/install/prefix" >&2
  exit 2
fi
package_path="$(realpath -- "$1")"
install_prefix="$(realpath -m -- "$2")"
case "${install_prefix}" in
  /usr|/usr/*|/opt|/opt/*|/|/usr/local/Ascend*)
    echo "Use a private installation prefix outside system directories" >&2
    exit 2
    ;;
esac
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  cann_prefix="$(realpath -- "${ASCEND_HOME_PATH}")"
  case "${install_prefix}" in
    "${cann_prefix}"|"${cann_prefix}"/*)
      echo "Refusing to install into the active CANN tree" >&2
      exit 2
      ;;
  esac
fi
mkdir -p -- "${install_prefix}"
bash "${package_path}" --quiet --install-path="${install_prefix}"
echo "Installed private operator package under ${install_prefix}"
