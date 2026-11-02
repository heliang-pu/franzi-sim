#!/usr/bin/env bash
# Start the stock-SDK simulation server with the Isaac Sim backend on the RTX 4090
# GPU workstation. Run from the project root on that machine:
#
#   ISAAC_PYTHON=<Isaac Sim python> sdk/run_isaacsim_4090.sh <new output dir> [--record] [--loop simulator] ...
#
# ISAAC_PYTHON is the interpreter of the Isaac Sim 5.1.0 pip install (see
# config/backends/isaacsim.yaml). Isaac keeps its own Python: ROS/conda library
# paths are dropped, as that machine requires. The NVIDIA Omniverse EULA must
# already be accepted for that install (isaacsim/kit/EULA_ACCEPTED, or
# OMNI_KIT_ACCEPT_EULA exported by the user); this script never accepts it.
# VK_ICD_FILENAMES keeps Vulkan to the driver package's ICD: a stale copy in
# /etc/vulkan/icd.d lists the one GPU twice, and Isaac Sim 5.1 then crashed
# (cuMemPoolSetAccess after the URDF import, 2026-10-24).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # the project root
PY="${ISAAC_PYTHON:?set ISAAC_PYTHON to the Isaac Sim interpreter}"
OUT="${1:?usage: $0 <new output dir> [server args]}"
shift
exec env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT -u PYTHONPATH \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
  "${PY}" -u -m simbridge.server --backend config/backends/isaacsim.yaml \
  --output "${OUT}" "$@"
