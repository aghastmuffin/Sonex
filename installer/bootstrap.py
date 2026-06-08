#!/usr/bin/env python3
"""
Sonex CLI installer (for scripts and automation).

For the graphical wizard, use setup_gui.py instead.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from installer.core import (
    DEFAULT_REF,
    InstallOptions,
    NullReporter,
    default_install_dir,
    find_python,
    run_install,
    venv_python,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sonex CLI installer")
    p.add_argument("--dir", type=Path, default=None, help="Install directory (default: ~/Sonex)")
    p.add_argument("--ref", default=DEFAULT_REF, help="Git ref: branch, tag, or 'latest'")
    p.add_argument("--with-mfa", action="store_true", help="Create conda 'mfa' environment")
    p.add_argument("--skip-download", action="store_true", help="Skip GitHub download")
    p.add_argument("--local", action="store_true", help="Copy from bundled/local checkout")
    p.add_argument("--skip-deps", action="store_true", help="Skip pip install")
    p.add_argument("--verify-only", action="store_true", help="Only run dependency checks")
    p.add_argument("--python", default=None, help="Python executable to use for venv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    install_dir = (args.dir or default_install_dir()).expanduser().resolve()
    reporter = NullReporter()

    if args.verify_only:
        from installer.core import run_checks
        py = venv_python(install_dir)
        warnings = run_checks(py if py.exists() else None, reporter)
        return 0 if not warnings else 1

    opts = InstallOptions(
        install_dir=install_dir,
        ref=args.ref,
        with_mfa=args.with_mfa,
        skip_download=args.skip_download,
        local=args.local,
        skip_deps=args.skip_deps,
        python_exe=args.python or find_python(),
    )

    result = run_install(opts, reporter)
    if not result.success:
        return 1
    if result.warnings:
        reporter.log("Some optional components are missing.")
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[error] Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        print(f"[error] Command failed: {' '.join(exc.cmd)}", file=sys.stderr)
        raise SystemExit(exc.returncode)
