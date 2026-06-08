#!/usr/bin/env bash
# Build Sonex Setup Wizard as a standalone app (macOS / Linux)
# Requires: pip install nuitka ordered-set zstandard

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="${SCRIPT_DIR}/dist"

python3 -m nuitka \
  --standalone \
  --enable-plugin=tk-inter \
  --output-dir="${OUT}" \
  --company-name="Sonex" \
  --product-name="Sonex Setup" \
  --include-module=installer.core \
  "${SCRIPT_DIR}/setup_gui.py"

echo "Built: ${OUT}/setup_gui.bin (or .app on macOS)"
