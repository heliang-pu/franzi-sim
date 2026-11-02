#!/usr/bin/env bash
# Start the stock-SDK simulation server with the Isaac Sim backend on the pro6000
# (RTX PRO 6000 Blackwell, NVIDIA driver 580). Run from the project root on that
# machine:
#
#   ISAAC_PYTHON=<Isaac Sim python> sdk/run_isaacsim_pro6000.sh <new output dir> [--record] [--loop simulator] ...
#
# ISAAC_PYTHON is the interpreter of that machine's Isaac Sim 5.1.0 pip install,
# whose EULA the user accepted (isaacsim/kit/EULA_ACCEPTED). Isaac Sim 5.1 does not
# render with the 595 driver the machine had before 2026-10-24. VK_ICD_FILENAMES
# keeps Vulkan to the driver package's ICD: a stale copy in /etc/vulkan/icd.d lists
# the one GPU twice, and Isaac Sim then crashes in multi-GPU mode.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # the project root
PY="${ISAAC_PYTHON:?set ISAAC_PYTHON to the Isaac Sim interpreter}"
OUT="${1:?usage: $0 <new output dir> [server args]}"
shift
exec env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT -u PYTHONPATH \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
  "${PY}" -u -m simbridge.server --backend config/backends/isaacsim.yaml \
  --output "${OUT}" "$@"
