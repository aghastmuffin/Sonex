"""Sonex install engine — shared by CLI bootstrap and GUI setup wizard."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import venv
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

OWNER = "aghastmuffin"
REPO = "Sonex"
DEFAULT_REF = "latest"
MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 12)
GITHUB_API = "https://api.github.com"

_SKIP_COPY = {
    ".git",
    ".venv",
    "__pycache__",
    "model_offload",
    "dist",
    ".DS_Store",
}


class ProgressReporter(Protocol):
    def status(self, message: str) -> None: ...
    def progress(self, percent: int, message: str = "") -> None: ...
    def log(self, message: str) -> None: ...


@dataclass
class InstallOptions:
    install_dir: Path
    ref: str = DEFAULT_REF
    with_mfa: bool = False
    skip_download: bool = False
    local: bool = False
    skip_deps: bool = False
    python_exe: str | None = None


@dataclass
class InstallResult:
    success: bool
    install_dir: Path
    python: Path | None = None
    warnings: list[str] = field(default_factory=list)
    error: str = ""


class NullReporter:
    def status(self, message: str) -> None:
        print(message)

    def progress(self, percent: int, message: str = "") -> None:
        if message:
            print(f"[{percent:3d}%] {message}")

    def log(self, message: str) -> None:
        print(message)


def default_install_dir() -> Path:
    override = os.environ.get("SONEX_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Sonex"
    return Path.home() / "Sonex"


def find_python(
    min_version: tuple[int, int] = MIN_PYTHON,
    max_version: tuple[int, int] = MAX_PYTHON,
) -> str:
    if getattr(sys, "frozen", False):
        return sys.executable

    candidates: list[str] = []
    for name in ("python3.12", "python3.11", "python3.10", "python3.9", "python3", "python"):
        found = shutil.which(name)
        if found and found not in candidates:
            candidates.append(found)
    if sys.executable not in candidates:
        candidates.append(sys.executable)

    best: str | None = None
    best_ver: tuple[int, int] | None = None
    for exe in candidates:
        try:
            out = subprocess.check_output(
                [exe, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            major, minor = map(int, out.split())
            ver = (major, minor)
            if ver < min_version or ver > max_version:
                continue
            if best_ver is None or ver > best_ver:
                best, best_ver = exe, ver
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            continue

    if best:
        return best
    raise RuntimeError(
        f"Python {min_version[0]}.{min_version[1]}–{max_version[0]}.{max_version[1]} is required."
    )


def venv_python(install_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return install_dir / ".venv" / "Scripts" / "python.exe"
    return install_dir / ".venv" / "bin" / "python"


def _http_get(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "Sonex-Installer/1.0"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_json(url: str) -> dict | list:
    data = _http_get(url, headers=_github_headers())
    return json.loads(data.decode("utf-8"))


def resolve_download_url(ref: str) -> tuple[str, str]:
    if ref == "latest":
        try:
            release = _fetch_json(f"{GITHUB_API}/repos/{OWNER}/{REPO}/releases/latest")
            ref = release["tag_name"]
        except urllib.error.HTTPError:
            ref = "main"

    if ref in ("main", "master"):
        url = f"https://github.com/{OWNER}/{REPO}/archive/refs/heads/{ref}.zip"
        return url, f"{REPO}-{ref}"

    tag_url = f"https://github.com/{OWNER}/{REPO}/archive/refs/tags/{ref}.zip"
    branch_url = f"https://github.com/{OWNER}/{REPO}/archive/refs/heads/{ref}.zip"

    if re.match(r"^v?\d", ref) or "-b" in ref:
        return tag_url, f"{REPO}-{ref}"
    if "/" in ref:
        url = f"https://github.com/{OWNER}/{REPO}/archive/{ref}.zip"
        return url, f"{REPO}-{ref.split('/')[-1]}"
    return branch_url, f"{REPO}-{ref}"


def _download(
    url: str,
    dest: Path,
    reporter: ProgressReporter,
    pct_start: int,
    pct_end: int,
) -> None:
    reporter.status("Downloading Sonex from GitHub...")
    reporter.log(url)
    req = urllib.request.Request(url, headers={"User-Agent": "Sonex-Installer/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        chunk_size = 1024 * 256
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                frac = done / total
                pct = pct_start + int(frac * (pct_end - pct_start))
                mb = done // 1024 // 1024
                reporter.progress(pct, f"Downloading... {mb} MB")
            else:
                reporter.progress(pct_start + (pct_end - pct_start) // 2, "Downloading...")


def extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="sonex-extract-"))
    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(tmp)
        elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(tmp)
        else:
            raise RuntimeError(f"Unsupported archive format: {archive}")

        entries = [p for p in tmp.iterdir() if p.name != "__MACOSX"]
        if len(entries) != 1 or not entries[0].is_dir():
            raise RuntimeError(f"Unexpected archive layout in {archive}")

        src_root = entries[0]
        for item in src_root.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _copy_tree(src: Path, dest: Path, reporter: ProgressReporter | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in _SKIP_COPY:
            continue
        if reporter:
            reporter.log(f"Copying: {item.name}")
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(item, target)


def _bundle_root() -> Path | None:
    """When frozen (Nuitka), locate bundled project files next to the executable."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        for candidate in (exe_dir, exe_dir / "sonex-bundle", exe_dir.parent):
            if (candidate / "Sonex.py").exists():
                return candidate
    here = Path(__file__).resolve().parent.parent
    if (here / "Sonex.py").exists():
        return here
    return None


def install_from_local(install_dir: Path, reporter: ProgressReporter) -> None:
    src_root = _bundle_root()
    if not src_root or not (src_root / "Sonex.py").exists():
        raise RuntimeError("Bundled Sonex files not found.")
    reporter.status("Copying Sonex files...")
    reporter.progress(5, "Preparing files...")
    _copy_tree(src_root, install_dir, reporter)
    reporter.log(f"Installed from {src_root}")


def download_project(install_dir: Path, ref: str, reporter: ProgressReporter) -> None:
    url, _folder = resolve_download_url(ref)
    with tempfile.TemporaryDirectory(prefix="sonex-dl-") as tmp:
        archive = Path(tmp) / "sonex.zip"
        _download(url, archive, reporter, 2, 28)
        reporter.status("Extracting files...")
        reporter.progress(30, "Extracting archive...")
        extract_archive(archive, install_dir)
    reporter.log("Project files ready.")


def create_venv(install_dir: Path, python_exe: str, reporter: ProgressReporter) -> Path:
    venv_dir = install_dir / ".venv"
    reporter.status("Creating virtual environment...")
    reporter.progress(35, "Setting up Python environment...")
    if not venv_dir.exists():
        venv.create(venv_dir, with_pip=True, clear=True)
    reporter.log(f"Virtual environment: {venv_dir}")
    return venv_python(install_dir)


def pip_install(py: Path, install_dir: Path, reporter: ProgressReporter) -> None:
    req = install_dir / "installer" / "requirements.txt"
    if not req.exists():
        req = install_dir / "docs" / "requirements_unclean.txt"
    if not req.exists():
        raise RuntimeError(f"No requirements file found in {install_dir}")

    reporter.status("Installing Python packages...")
    reporter.progress(40, "Upgrading pip...")
    subprocess.check_call(
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    reporter.progress(45, "Installing dependencies (this may take several minutes)...")
    proc = subprocess.Popen(
        [str(py), "-m", "pip", "install", "-r", str(req)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    line_count = 0
    for line in proc.stdout:
        line = line.strip()
        if line:
            reporter.log(line[:72])
            line_count += 1
            pct = min(84, 45 + line_count // 3)
            reporter.progress(pct, "Installing packages...")
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"pip install failed (exit {code})")
    reporter.progress(85, "Packages installed.")
    reporter.log("Python dependencies installed.")


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def setup_mfa_env(reporter: ProgressReporter) -> None:
    conda = _which("conda")
    if not conda:
        raise RuntimeError("conda is required for MFA setup.")

    reporter.status("Creating MFA alignment environment...")
    reporter.progress(86, "Installing Montreal Forced Aligner...")
    subprocess.check_call(
        [conda, "create", "-n", "mfa", "-y", "-c", "conda-forge", "montreal-forced-aligner"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    reporter.progress(92, "MFA environment ready.")
    reporter.log("Conda environment 'mfa' created.")


def write_launchers(install_dir: Path, py: Path, reporter: ProgressReporter) -> None:
    reporter.status("Creating shortcuts...")
    reporter.progress(93, "Writing launch scripts...")
    bin_dir = install_dir / "bin"
    bin_dir.mkdir(exist_ok=True)
    sonex_py = install_dir / "Sonex.py"

    if sys.platform.startswith("win"):
        for name, extra in (("sonex", ""), ("sonex-view", "--view")):
            (bin_dir / f"{name}.bat").write_text(
                f'@echo off\r\nsetlocal\r\ncd /d "{install_dir}"\r\n'
                f'"{py}" "{sonex_py}" {extra} %*\r\n',
                encoding="utf-8",
            )
        verify_py = install_dir / "installer" / "verify.py"
        (bin_dir / "sonex-verify.bat").write_text(
            f'@echo off\r\n"{py}" "{verify_py}" %*\r\n',
            encoding="utf-8",
        )
        # Desktop shortcut-style launcher
        (bin_dir / "Sonex.lnk.bat").write_text(
            f'@echo off\r\nstart "" "{bin_dir / "sonex.bat"}"\r\n',
            encoding="utf-8",
        )
    else:
        for name, extra in (("sonex", ""), ("sonex-view", "--view")):
            script = bin_dir / name
            script.write_text(
                f'#!/usr/bin/env bash\nset -euo pipefail\n'
                f'cd "{install_dir}"\nexec "{py}" "{sonex_py}" {extra} "$@"\n',
                encoding="utf-8",
            )
            script.chmod(0o755)

    reporter.log(f"Launchers written to {bin_dir}")


def write_env_file(install_dir: Path) -> None:
    env_file = install_dir / "sonex.env"
    lines = [
        f"SONEX_HOME={install_dir}",
        f"SONEX_PYTHON={venv_python(install_dir)}",
    ]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_checks(py: Path | None, reporter: ProgressReporter) -> list[str]:
    warnings: list[str] = []
    reporter.status("Verifying installation...")
    reporter.progress(96, "Running checks...")

    for tool, label in (("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")):
        if _which(tool):
            reporter.log(f"  [OK] {label}")
        else:
            msg = f"{label} not found"
            warnings.append(msg)
            reporter.log(f"  [!!] {msg}")

    conda = _which("conda")
    if conda:
        reporter.log("  [OK] conda")
        try:
            out = subprocess.check_output([conda, "env", "list", "--json"], stderr=subprocess.DEVNULL, text=True)
            envs = json.loads(out).get("envs", [])
            if any(Path(e).name == "mfa" for e in envs):
                reporter.log("  [OK] mfa environment")
            else:
                warnings.append("MFA conda env not found (phone alignment unavailable)")
                reporter.log("  [!!] mfa environment missing")
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            warnings.append("Could not verify MFA environment")
    else:
        warnings.append("conda not found (MFA alignment unavailable)")
        reporter.log("  [!!] conda not found")

    if py and py.exists():
        for mod, label in [
            ("PyQt6", "PyQt6"),
            ("demucs", "demucs"),
            ("faster_whisper", "faster-whisper"),
            ("numpy", "numpy"),
        ]:
            try:
                subprocess.check_call(
                    [str(py), "-c", f"import {mod}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                reporter.log(f"  [OK] {label}")
            except subprocess.CalledProcessError:
                warnings.append(f"{label} import failed")
                reporter.log(f"  [!!] {label}")

    reporter.progress(100, "Complete.")
    return warnings


def run_install(opts: InstallOptions, reporter: ProgressReporter | None = None) -> InstallResult:
    rep = reporter or NullReporter()
    install_dir = opts.install_dir.expanduser().resolve()
    warnings: list[str] = []

    try:
        python_exe = opts.python_exe or find_python()
        rep.log(f"Using Python: {python_exe}")
        rep.log(f"Install location: {install_dir}")

        install_dir.mkdir(parents=True, exist_ok=True)
        rep.progress(1, "Starting setup...")

        bundled = _bundle_root()
        use_bundle = opts.local or (opts.skip_download and bundled is not None)

        if use_bundle:
            install_from_local(install_dir, rep)
        elif not opts.skip_download:
            download_project(install_dir, opts.ref, rep)
        elif not (install_dir / "Sonex.py").exists():
            if bundled:
                install_from_local(install_dir, rep)
            else:
                raise RuntimeError("No Sonex files found. Cannot continue.")

        py = create_venv(install_dir, python_exe, rep)

        if not opts.skip_deps:
            pip_install(py, install_dir, rep)

        if opts.with_mfa:
            try:
                setup_mfa_env(rep)
            except Exception as exc:
                warnings.append(f"MFA setup failed: {exc}")
                rep.log(f"MFA setup skipped: {exc}")

        write_launchers(install_dir, py, rep)
        write_env_file(install_dir)
        warnings.extend(run_checks(py, rep))

        rep.status("Installation complete!")
        return InstallResult(success=True, install_dir=install_dir, python=py, warnings=warnings)

    except Exception as exc:
        rep.log(f"ERROR: {exc}")
        rep.status("Installation failed.")
        return InstallResult(success=False, install_dir=install_dir, error=str(exc), warnings=warnings)
