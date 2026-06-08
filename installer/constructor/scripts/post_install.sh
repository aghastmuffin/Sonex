#!/usr/bin/env bash
# Post-install hook for constructor-built packages.
echo "Sonex installed. Run 'sonex' from your install bin directory."
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg: OK"
else
  echo "ffmpeg: not found — install via your package manager or conda."
fi
if command -v conda >/dev/null 2>&1 && conda env list | grep -q '\bmfa\b'; then
  echo "MFA conda env: OK"
else
  echo "MFA conda env: not found — run: conda env create -f installer/environment-mfa.yml"
fi
