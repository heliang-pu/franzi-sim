#!/usr/bin/env bash
# Run a module of the Isaac Sim port under Isaac Sim's own Python, from the
# project root:
#
#   ISAAC_PYTHON=<Isaac Sim python> isaac/run.sh isaac.check outputs/<dir>/<scene>
#
# The environment of sdk/run_isaacsim_*.sh: ROS/conda library paths are dropped,
# and VK_ICD_FILENAMES keeps Vulkan to the driver package's ICD (a stale copy in
# /etc/vulkan/icd.d lists the one GPU twice).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # the project root
PY="${ISAAC_PYTHON:?set ISAAC_PYTHON to the Isaac Sim interpreter}"
exec env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT -u PYTHONPATH \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
  "${PY}" -u -m "$@"
