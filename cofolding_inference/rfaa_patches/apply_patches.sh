#!/usr/bin/env bash
# Apply KinConfBench patches to a RoseTTAFold-All-Atom clone.
# Usage: bash apply_patches.sh [RFAA_REPO]   (defaults to $RFAA_REPO)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-${RFAA_REPO:-}}"
if [ -z "$REPO" ]; then
  echo "Set RFAA_REPO or pass the repo path as arg 1" >&2
  exit 1
fi
python "$HERE/apply_patches.py" "$REPO"
