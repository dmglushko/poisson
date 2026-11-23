#!/bin/bash
# Build the stage programs on DelftBlue. Run on a login node, once and after each C++ change:
#
#     bash scripts/delftblue/build.sh            # Release build in build-delftblue/
#     JOBS=8 bash scripts/delftblue/build.sh     # override parallel compile jobs (default: 4)
#
# Every job uses this one build. Rebuilding changes the stage binaries, and so their cache keys:
# don't rebuild while an experiment is queued or running.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

cmake -S "${POISSON_REPO}" -B "${POISSON_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${POISSON_BUILD_DIR}" -j "${JOBS:-4}"

echo
echo "Built into ${POISSON_BUILD_DIR}"
