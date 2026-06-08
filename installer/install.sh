#!/usr/bin/env bash
# Sonex one-line installer for macOS / Linux
#
#   curl -fsSL https://raw.githubusercontent.com/aghastmuffin/Sonex/main/installer/install.sh | bash
#
# Options (env vars):
#   SONEX_HOME=/path/to/install   Install directory (default: ~/Sonex)
#   SONEX_REF=0.4-b               Git tag/branch to install (default: main)
#   SONEX_WITH_MFA=1              Also create conda MFA environment
set -euo pipefail

OWNER="aghastmuffin"
REPO="Sonex"
REF="${SONEX_REF:-main}"
RAW="https://raw.githubusercontent.com/${OWNER}/${REPO}/${REF}/installer"

echo "==> Sonex installer"
echo "    ref: ${REF}"

find_python() {
  for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
      ver=$("$cmd" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)
      major=${ver%%.*}
      minor=${ver#*.}
      if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 9 ]; }; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  echo "ERROR: Python 3.9+ is required." >&2
  echo "Install Python from https://www.python.org/downloads/ or your package manager." >&2
  exit 1
}

PYTHON=$(find_python)
echo "    python: $PYTHON ($($PYTHON --version 2>&1))"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

curl -fsSL "${RAW}/bootstrap.py" -o "${TMP}/bootstrap.py"

ARGS=(--ref "$REF")
if [ -n "${SONEX_HOME:-}" ]; then
  ARGS+=(--dir "$SONEX_HOME")
fi
if [ "${SONEX_WITH_MFA:-0}" = "1" ]; then
  ARGS+=(--with-mfa)
fi

exec "$PYTHON" "${TMP}/bootstrap.py" "${ARGS[@]}"
