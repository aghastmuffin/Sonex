#!/usr/bin/env python3
"""
Sonex Installer
Cross-platform setup script for Mac / Ubuntu / Windows 10/11.

What it does:
  1. Creates a Python virtual environment (.venv) in the project root using
     uv (fast) with a fallback to the stdlib 'venv' module.
  2. Upgrades pip inside that venv.
  3. Installs all pip packages listed in requirements.txt into the venv.
  4. Creates a *separate* conda/mamba environment (mfa-env/) in the project
     root and installs montreal-forced-aligner from conda-forge inside it.

All subprocess output is streamed live into the Tkinter GUI log window.
"""

import os
import platform
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VENV_DIR     = os.path.join(PROJECT_ROOT, ".venv")
MFA_ENV_DIR  = os.path.join(PROJECT_ROOT, "mfa-env")
REQUIREMENTS = os.path.join(SCRIPT_DIR, "requirements.txt")

IS_WINDOWS = platform.system() == "Windows"

_bin = "Scripts" if IS_WINDOWS else "bin"
VENV_PYTHON = os.path.join(VENV_DIR, _bin, "python" + (".exe" if IS_WINDOWS else ""))
VENV_PIP    = os.path.join(VENV_DIR, _bin, "pip"    + (".exe" if IS_WINDOWS else ""))

# Total logical steps used to calculate progress bar percentage
TOTAL_STEPS = 5


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class InstallerGUI:
    """Minimal Tkinter installer window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Sonex Installer")
        root.geometry("720x520")
        root.resizable(True, True)

        tk.Label(root, text="Sonex Installer", font=("Helvetica", 18, "bold")).pack(pady=(16, 2))
        tk.Label(root, text="Setting up your environment…", font=("Helvetica", 11)).pack(pady=(0, 10))

        self.status_var = tk.StringVar(value="Initializing…")
        tk.Label(root, textvariable=self.status_var, font=("Helvetica", 10), anchor="w").pack(
            fill="x", padx=20
        )

        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            root,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=680,
        ).pack(padx=20, pady=(4, 10))

        self.log_area = scrolledtext.ScrolledText(
            root, height=18, state="disabled", font=("Courier", 9)
        )
        self.log_area.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        self.close_btn = tk.Button(
            root, text="Close", state="disabled", command=root.destroy, width=14
        )
        self.close_btn.pack(pady=(0, 12))

    # ------------------------------------------------------------------
    # Thread-safe helpers
    # ------------------------------------------------------------------
    def log(self, text: str) -> None:
        self.root.after(0, self._append_log, text)

    def _append_log(self, text: str) -> None:
        self.log_area.config(state="normal")
        self.log_area.insert("end", text + "\n")
        self.log_area.see("end")
        self.log_area.config(state="disabled")

    def set_status(self, text: str) -> None:
        self.root.after(0, self.status_var.set, text)

    def set_progress(self, value: float) -> None:
        self.root.after(0, self.progress_var.set, value)

    def finish(self, success: bool = True) -> None:
        def _finish() -> None:
            self.close_btn.config(state="normal")
            if success:
                self.status_var.set("✓ Installation complete!")
                self.progress_var.set(100)
            else:
                self.status_var.set("✗ Installation failed — see log above.")

        self.root.after(0, _finish)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_cmd(cmd, gui: InstallerGUI, cwd: str = None, env=None) -> bool:
    """Run *cmd* (list or string), stream every output line to *gui*, return success."""
    display = " ".join(cmd) if isinstance(cmd, list) else cmd
    gui.log(f"$ {display}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=env,
            shell=isinstance(cmd, str),
        )
        for line in proc.stdout:
            stripped = line.rstrip()
            if stripped:
                gui.log(stripped)
        proc.wait()
        if proc.returncode != 0:
            gui.log(f"[ERROR] exited with code {proc.returncode}")
            return False
        return True
    except FileNotFoundError as exc:
        gui.log(f"[ERROR] executable not found: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        gui.log(f"[ERROR] {exc}")
        return False


def find_conda():
    """Return (path, name) for the best available conda/mamba executable."""
    for name in ("mamba", "conda"):
        path = shutil.which(name)
        if path:
            return path, name

    home = os.path.expanduser("~")
    if IS_WINDOWS:
        candidates = [
            os.path.join(home, "mambaforge",  "Scripts", "mamba.exe"),
            os.path.join(home, "miniforge3",  "Scripts", "mamba.exe"),
            os.path.join(home, "miniconda3",  "Scripts", "conda.exe"),
            os.path.join(home, "anaconda3",   "Scripts", "conda.exe"),
            r"C:\ProgramData\mambaforge\Scripts\mamba.exe",
            r"C:\ProgramData\miniforge3\Scripts\mamba.exe",
            r"C:\ProgramData\miniconda3\Scripts\conda.exe",
        ]
    else:
        candidates = [
            os.path.join(home, "mambaforge",  "bin", "mamba"),
            os.path.join(home, "miniforge3",  "bin", "mamba"),
            os.path.join(home, "micromamba",  "bin", "micromamba"),
            os.path.join(home, "miniconda3",  "bin", "conda"),
            os.path.join(home, "anaconda3",   "bin", "conda"),
            "/opt/homebrew/bin/mamba",
            "/usr/local/bin/mamba",
            "/opt/conda/bin/conda",
        ]

    for path in candidates:
        if os.path.isfile(path):
            name = os.path.splitext(os.path.basename(path))[0]
            return path, name

    return None, None


# ---------------------------------------------------------------------------
# Installation logic (runs in a background thread)
# ---------------------------------------------------------------------------
def install(gui: InstallerGUI) -> None:
    step = 0
    conda_exe = None  # populated in step 4, used in summary

    def progress(n: int) -> None:
        # Reserve the final 10 % for the summary step
        gui.set_progress(min(n / TOTAL_STEPS * 90, 90))

    # ------------------------------------------------------------------ #
    # Step 1 — create virtual environment                                  #
    # ------------------------------------------------------------------ #
    step += 1
    gui.set_status(f"[{step}/{TOTAL_STEPS}] Creating Python virtual environment…")
    progress(step)
    gui.log("\n=== Step 1: Create virtual environment ===")

    uv = shutil.which("uv")
    venv_ok = False

    if uv:
        gui.log("uv detected — using it to create the venv (faster).")
        venv_ok = run_cmd([uv, "venv", VENV_DIR], gui)

    if not venv_ok:
        if uv:
            gui.log("uv venv creation failed; falling back to python -m venv.")
        else:
            gui.log("uv not found; using python -m venv.")
        venv_ok = run_cmd([sys.executable, "-m", "venv", VENV_DIR], gui)

    if not venv_ok:
        gui.log("[FATAL] Could not create virtual environment. Aborting.")
        gui.finish(success=False)
        return

    # ------------------------------------------------------------------ #
    # Step 2 — upgrade pip                                                 #
    # ------------------------------------------------------------------ #
    step += 1
    gui.set_status(f"[{step}/{TOTAL_STEPS}] Upgrading pip…")
    progress(step)
    gui.log("\n=== Step 2: Upgrade pip ===")
    run_cmd([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"], gui)

    # ------------------------------------------------------------------ #
    # Step 3 — install pip packages                                        #
    # ------------------------------------------------------------------ #
    step += 1
    gui.set_status(f"[{step}/{TOTAL_STEPS}] Installing Python packages (may take several minutes)…")
    progress(step)
    gui.log("\n=== Step 3: Install packages from requirements.txt ===")

    pip_ok = False
    if uv:
        gui.log("Using uv pip install (faster).")
        pip_ok = run_cmd(
            [uv, "pip", "install", "--python", VENV_PYTHON, "-r", REQUIREMENTS],
            gui,
        )
        if not pip_ok:
            gui.log("uv pip install failed; falling back to pip.")

    if not pip_ok:
        pip_ok = run_cmd([VENV_PIP, "install", "-r", REQUIREMENTS], gui)

    if not pip_ok:
        gui.log("[WARNING] Some packages may not have been installed correctly.")

    # ------------------------------------------------------------------ #
    # Step 4 — conda/mamba environment for MFA                            #
    # ------------------------------------------------------------------ #
    step += 1
    gui.set_status(f"[{step}/{TOTAL_STEPS}] Setting up MFA conda environment…")
    progress(step)
    gui.log("\n=== Step 4: Create MFA conda environment ===")

    conda_exe, conda_name = find_conda()

    if conda_exe is None:
        gui.log(
            "[WARNING] conda/mamba not found on this system.\n"
            "          Montreal Forced Aligner was NOT installed.\n"
            "          To install it manually:\n"
            "            1. Install miniforge3 or miniconda3\n"
            f"           2. conda create -p {MFA_ENV_DIR} python=3.10 -y\n"
            f"           3. conda install -p {MFA_ENV_DIR} -c conda-forge montreal-forced-aligner -y"
        )
    else:
        gui.log(f"Found {conda_name} at: {conda_exe}")
        gui.log(f"Creating conda env at {MFA_ENV_DIR} …")

        create_ok = run_cmd(
            [conda_exe, "create", "-p", MFA_ENV_DIR, "python=3.10", "-y"],
            gui,
        )

        if create_ok:
            gui.log("Installing montreal-forced-aligner from conda-forge…")
            mfa_ok = run_cmd(
                [
                    conda_exe, "install",
                    "-p", MFA_ENV_DIR,
                    "-c", "conda-forge",
                    "montreal-forced-aligner",
                    "-y",
                ],
                gui,
            )
            if mfa_ok:
                gui.log("MFA installed successfully.")
            else:
                gui.log(
                    "[WARNING] MFA installation failed.\n"
                    "          You can retry manually:\n"
                    f"          conda install -p {MFA_ENV_DIR} -c conda-forge montreal-forced-aligner -y"
                )
        else:
            gui.log(
                "[WARNING] Could not create MFA conda environment.\n"
                "          Check that conda/mamba is properly initialized."
            )

    # ------------------------------------------------------------------ #
    # Step 5 — summary                                                     #
    # ------------------------------------------------------------------ #
    step += 1
    gui.set_status(f"[{step}/{TOTAL_STEPS}] Done!")
    gui.set_progress(100)
    gui.log("\n=== Installation Summary ===")
    gui.log(f"  Project venv  : {VENV_DIR}")
    gui.log(f"  MFA conda env : {MFA_ENV_DIR}")

    if IS_WINDOWS:
        activate = os.path.join(VENV_DIR, "Scripts", "Activate.ps1")
        gui.log(f"  Activate venv : {activate}  (PowerShell)")
    else:
        activate = os.path.join(VENV_DIR, "bin", "activate")
        gui.log(f"  Activate venv : source {activate}")

    if conda_exe:
        gui.log(f"  Run MFA       : {conda_exe} run -p {MFA_ENV_DIR} mfa --help")

    gui.log("")
    gui.log("All done! You can close this window.")
    gui.finish(success=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    root = tk.Tk()
    gui = InstallerGUI(root)
    # Run installation in a daemon thread so the GUI event loop is never blocked
    threading.Thread(target=install, args=(gui,), daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
