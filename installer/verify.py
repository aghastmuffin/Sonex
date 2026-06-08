#!/usr/bin/env python3
"""Post-install verification for Sonex."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _venv_python(root: Path) -> Path:
    if sys.platform.startswith("win"):
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _check(cmd: list[str], label: str) -> bool:
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  OK  {label}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  FAIL {label}")
        return False


def main() -> int:
    root = Path(os.environ.get("SONEX_HOME", _root())).resolve()
    py = Path(os.environ.get("SONEX_PYTHON", _venv_python(root)))

    print(f"Sonex verify — {root}")
    print()

    ok = True

    print("Project layout:")
    for rel in ("Sonex.py", "ui/_worker.py", "backbone/ltra/letra_toolkit.py", "installer/requirements.txt"):
        path = root / rel
        if path.exists():
            print(f"  OK  {rel}")
        else:
            print(f"  FAIL {rel}")
            ok = False

    print()
    print("Python environment:")
    if not py.exists():
        print(f"  FAIL venv python not found at {py}")
        ok = False
    else:
        print(f"  OK  {py}")
        for mod, label in [
            ("PyQt6", "PyQt6"),
            ("demucs", "demucs"),
            ("faster_whisper", "faster-whisper"),
            ("librosa", "librosa"),
            ("numpy", "numpy"),
            ("praatio", "praatio"),
            ("argostranslate", "argostranslate"),
        ]:
            if not _check([str(py), "-c", f"import {mod}"], label):
                ok = False

    print()
    print("External tools:")
    ok &= _check(["ffmpeg", "-version"], "ffmpeg")
    ok &= _check(["ffprobe", "-version"], "ffprobe")

    try:
        subprocess.check_call(
            ["conda", "run", "-n", "mfa", "mfa", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("  OK  mfa (conda env)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  WARN mfa conda env (optional — needed for phone-level alignment)")

    print()
    if ok:
        print("All required checks passed.")
        return 0
    print("Some required checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
