#!/usr/bin/env bash
INSTALL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$INSTALL_DIR/sonex-app" 2>/dev/null || cd "$INSTALL_DIR"
exec python Sonex.py --view "$@"
