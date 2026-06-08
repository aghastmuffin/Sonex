#!/usr/bin/env python3
"""
Sonex Setup Wizard — retro Windows 9x style GUI installer.

Double-click to run. Intended for Nuitka packaging (--windows-disable-console).

  python installer/setup_gui.py
  nuitka --standalone --windows-disable-console --enable-plugin=tk-inter installer/setup_gui.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

# Allow running as script or frozen bundle
_INSTALLER_DIR = Path(__file__).resolve().parent
if str(_INSTALLER_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_INSTALLER_DIR.parent))

from installer.core import (  # noqa: E402
    DEFAULT_REF,
    InstallOptions,
    ProgressReporter,
    default_install_dir,
    run_install,
)

# ---------------------------------------------------------------------------
# Classic Windows 9x palette (NVIDIA-driver-wizard energy)
# ---------------------------------------------------------------------------

WIN_BG = "#C0C0C0"
WIN_FACE = "#C0C0C0"
WIN_HIGHLIGHT = "#000080"
WIN_TEXT = "#000000"
WIN_DISABLED = "#808080"
TROUGH_BG = "#E8E4D9"       # off-white / beige trough
TROUGH_BORDER_DARK = "#404040"
TROUGH_BORDER_LIGHT = "#FFFFFF"
SEGMENT_FILL = "#00A000"    # NVIDIA green
SEGMENT_FILL_HI = "#00D800"
SEGMENT_EMPTY = "#D4D0C8"
SEGMENT_BORDER = "#606060"
GROUP_BG = "#C0C0C0"
LOG_BG = "#FFFFFF"
LOG_FG = "#000080"
TITLE_BG = "#000080"
TITLE_FG = "#FFFFFF"
BANNER_BG = "#000080"


class SegmentedProgressBar(tk.Canvas):
    """Blocky segmented progress bar — green boxes on an off-white trough."""

    def __init__(self, master, segments: int = 28, height: int = 22, **kwargs):
        super().__init__(
            master,
            height=height,
            highlightthickness=0,
            borderwidth=0,
            bg=WIN_BG,
            **kwargs,
        )
        self.segments = segments
        self._filled = 0
        self._pulse = 0
        self._animating = False
        self.bind("<Configure>", self._redraw)

    def set_percent(self, percent: int) -> None:
        filled = max(0, min(self.segments, round(percent / 100 * self.segments)))
        if filled != self._filled:
            self._filled = filled
            self._redraw()

    def set_indeterminate(self, active: bool) -> None:
        self._animating = active
        if active:
            self._pulse_tick()
        else:
            self._redraw()

    def _pulse_tick(self) -> None:
        if not self._animating:
            return
        self._pulse = (self._pulse + 1) % (self.segments + 4)
        self._redraw()
        self.after(120, self._pulse_tick)

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        w = max(self.winfo_width(), 200)
        h = max(self.winfo_height(), 18)
        pad = 2

        # Sunken trough
        self.create_rectangle(0, 0, w, h, fill=TROUGH_BORDER_DARK, outline="")
        self.create_rectangle(1, 1, w - 1, h - 1, fill=TROUGH_BORDER_LIGHT, outline="")
        self.create_rectangle(2, 2, w - 2, h - 2, fill=TROUGH_BG, outline="")

        inner_w = w - 6
        inner_h = h - 6
        gap = 2
        seg_w = max(4, (inner_w - gap * (self.segments - 1)) // self.segments)
        x0 = 3
        y0 = 3

        for i in range(self.segments):
            x1 = x0 + i * (seg_w + gap)
            x2 = x1 + seg_w
            y1, y2 = y0, y0 + inner_h

            if self._animating:
                # Scrolling highlight blocks during indeterminate phases
                pulse_start = self._pulse
                filled = pulse_start <= i < pulse_start + 4
            else:
                filled = i < self._filled

            if filled:
                # Raised green block
                self.create_rectangle(x1, y1, x2, y2, fill=SEGMENT_BORDER, outline="")
                self.create_rectangle(x1 + 1, y1 + 1, x2 - 1, y2 - 1, fill=SEGMENT_FILL, outline="")
                self.create_line(x1 + 1, y1 + 1, x2 - 2, y1 + 1, fill=SEGMENT_FILL_HI)
            else:
                self.create_rectangle(x1, y1, x2, y2, fill=SEGMENT_BORDER, outline="")
                self.create_rectangle(x1 + 1, y1 + 1, x2 - 1, y2 - 1, fill=SEGMENT_EMPTY, outline="")


def _win95_button(master, text: str, command, width: int = 10, default: bool = False) -> tk.Button:
    return tk.Button(
        master,
        text=text,
        command=command,
        width=width,
        font=("MS Sans Serif", 8),
        bg=WIN_FACE,
        fg=WIN_TEXT,
        activebackground="#DFDFDF",
        activeforeground=WIN_TEXT,
        relief=tk.RAISED,
        bd=2,
        default=tk.ACTIVE if default else tk.NORMAL,
        cursor="hand2",
    )


def _win95_entry(master, textvariable: tk.StringVar, width: int = 42) -> tk.Entry:
    return tk.Entry(
        master,
        textvariable=textvariable,
        width=width,
        font=("MS Sans Serif", 8),
        bg=LOG_BG,
        fg=WIN_TEXT,
        relief=tk.SUNKEN,
        bd=2,
    )


def _group_frame(master, title: str) -> tk.Frame:
    outer = tk.Frame(master, bg=GROUP_BG, relief=tk.GROOVE, bd=2)
    lbl = tk.Label(
        outer,
        text=f" {title} ",
        font=("MS Sans Serif", 8, "bold"),
        bg=GROUP_BG,
        fg=WIN_TEXT,
    )
    lbl.place(x=8, y=-2)
    return outer


class GuiReporter:
    def __init__(self, app: "SonexSetupWizard"):
        self.app = app

    def status(self, message: str) -> None:
        self.app._ui(self.app._set_status, message)

    def progress(self, percent: int, message: str = "") -> None:
        self.app._ui(self.app._set_progress, percent, message or None)

    def log(self, message: str) -> None:
        self.app._ui(self.app._append_log, message)


class SonexSetupWizard:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Sonex Setup")
        self.root.configure(bg=WIN_BG)
        self.root.resizable(False, False)

        # Classic fixed wizard size
        self.root.geometry("500x380")
        self._center_window()

        self.install_dir = tk.StringVar(value=str(default_install_dir()))
        self.with_mfa = tk.BooleanVar(value=False)
        self.ref = tk.StringVar(value=DEFAULT_REF)

        self._page: str = "welcome"
        self._install_thread: threading.Thread | None = None
        self._result = None

        self._build_chrome()
        self._show_welcome()

        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _center_window(self) -> None:
        self.root.update_idletasks()
        w, h = 500, 380
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_chrome(self) -> None:
        # Title banner (Win9x installer header strip)
        self.banner = tk.Frame(self.root, bg=BANNER_BG, height=46)
        self.banner.pack(fill=tk.X)
        self.banner.pack_propagate(False)

        tk.Label(
            self.banner,
            text="Sonex",
            font=("MS Sans Serif", 14, "bold"),
            bg=BANNER_BG,
            fg=TITLE_FG,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(12, 0), pady=6)

        tk.Label(
            self.banner,
            text="Setup Wizard",
            font=("MS Sans Serif", 9),
            bg=BANNER_BG,
            fg="#B0C4FF",
            anchor="w",
        ).pack(side=tk.LEFT, padx=(8, 0), pady=10)

        # Main content area
        self.content = tk.Frame(self.root, bg=WIN_BG)
        self.content.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Button bar (always at bottom)
        self.btnbar = tk.Frame(self.root, bg=WIN_BG)
        self.btnbar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))

        self.btn_back = _win95_button(self.btnbar, "< Back", self._on_back, width=9)
        self.btn_back.pack(side=tk.LEFT)

        self.btn_next = _win95_button(self.btnbar, "Next >", self._on_next, width=9, default=True)
        self.btn_next.pack(side=tk.RIGHT, padx=(4, 0))

        self.btn_cancel = _win95_button(self.btnbar, "Cancel", self._on_cancel, width=9)
        self.btn_cancel.pack(side=tk.RIGHT)

        # Win9x grooved separator above button bar
        sep = tk.Frame(self.root, bg=WIN_BG, height=2, relief=tk.GROOVE, bd=1)
        sep.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 2))

    def _clear_content(self) -> None:
        for w in self.content.winfo_children():
            w.destroy()

    def _show_welcome(self) -> None:
        self._page = "welcome"
        self._clear_content()
        self.btn_back.config(state=tk.DISABLED)
        self.btn_next.config(text="Next >", state=tk.NORMAL)
        self.btn_cancel.config(state=tk.NORMAL)

        left = tk.Frame(self.content, bg=WIN_BG)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        # Fake CD-ROM / box art panel
        art = tk.Canvas(left, width=80, height=120, bg=WIN_BG, highlightthickness=0)
        art.pack()
        art.create_rectangle(4, 4, 76, 116, fill="#808080", outline="#404040")
        art.create_rectangle(8, 8, 72, 112, fill="#A0A0A0", outline="")
        art.create_text(40, 50, text="SONEX", fill="#000080", font=("MS Sans Serif", 10, "bold"))
        art.create_text(40, 72, text="v0.4", fill=WIN_TEXT, font=("MS Sans Serif", 7))

        right = tk.Frame(self.content, bg=WIN_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            right,
            text="Welcome to the Sonex Setup Wizard",
            font=("MS Sans Serif", 9, "bold"),
            bg=WIN_BG,
            fg=WIN_TEXT,
            anchor="w",
        ).pack(fill=tk.X, pady=(4, 8))

        tk.Label(
            right,
            text=(
                "This wizard will install Sonex on your computer.\n\n"
                "Setup will download the latest files from GitHub,\n"
                "install required components, and create program\n"
                "shortcuts in your Start menu folder.\n\n"
                "WARNING: This program is protected by copyright\n"
                "law and international treaties. Unauthorized\n"
                "reproduction or distribution may result in\n"
                "severe civil and criminal penalties.\n\n"
                "Click Next to continue, or Cancel to exit Setup."
            ),
            font=("MS Sans Serif", 8),
            bg=WIN_BG,
            fg=WIN_TEXT,
            justify=tk.LEFT,
            anchor="nw",
        ).pack(fill=tk.BOTH, expand=True)

    def _show_options(self) -> None:
        self._page = "options"
        self._clear_content()
        self.btn_back.config(state=tk.NORMAL)
        self.btn_next.config(text="Install", state=tk.NORMAL)

        tk.Label(
            self.content,
            text="Select Installation Options",
            font=("MS Sans Serif", 9, "bold"),
            bg=WIN_BG,
            fg=WIN_TEXT,
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 8))

        grp = _group_frame(self.content, "Destination Folder")
        grp.pack(fill=tk.X, pady=(6, 10), ipady=14, ipadx=6)

        row = tk.Frame(grp, bg=GROUP_BG)
        row.pack(fill=tk.X, padx=10, pady=(12, 6))

        tk.Label(row, text="Install Sonex to:", font=("MS Sans Serif", 8), bg=GROUP_BG).pack(anchor="w")
        path_row = tk.Frame(row, bg=GROUP_BG)
        path_row.pack(fill=tk.X, pady=(4, 0))
        _win95_entry(path_row, self.install_dir, width=44).pack(side=tk.LEFT)
        _win95_button(path_row, "Browse...", self._browse_dir, width=9).pack(side=tk.LEFT, padx=(6, 0))

        grp2 = _group_frame(self.content, "Components")
        grp2.pack(fill=tk.X, pady=(0, 6), ipady=10, ipadx=6)

        inner = tk.Frame(grp2, bg=GROUP_BG)
        inner.pack(fill=tk.X, padx=10, pady=(12, 4))

        tk.Checkbutton(
            inner,
            text="Install MFA alignment environment (requires conda)",
            variable=self.with_mfa,
            font=("MS Sans Serif", 8),
            bg=GROUP_BG,
            activebackground=GROUP_BG,
        ).pack(anchor="w")

        tk.Label(
            inner,
            text=f"Release channel: {self.ref.get()}  (from GitHub)",
            font=("MS Sans Serif", 8),
            bg=GROUP_BG,
            fg=WIN_DISABLED,
        ).pack(anchor="w", pady=(6, 0))

    def _show_installing(self) -> None:
        self._page = "installing"
        self._clear_content()
        self.btn_back.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.DISABLED)

        tk.Label(
            self.content,
            text="Installing Sonex",
            font=("MS Sans Serif", 9, "bold"),
            bg=WIN_BG,
            fg=WIN_TEXT,
            anchor="w",
        ).pack(fill=tk.X)

        tk.Label(
            self.content,
            text="Please wait while Setup installs Sonex on your computer.",
            font=("MS Sans Serif", 8),
            bg=WIN_BG,
            fg=WIN_TEXT,
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 10))

        prog_grp = _group_frame(self.content, "Setup Progress")
        prog_grp.pack(fill=tk.X, pady=(0, 8), ipady=12, ipadx=6)

        prog_inner = tk.Frame(prog_grp, bg=GROUP_BG)
        prog_inner.pack(fill=tk.X, padx=10, pady=(14, 6))

        self.status_label = tk.Label(
            prog_inner,
            text="Preparing installation...",
            font=("MS Sans Serif", 8),
            bg=GROUP_BG,
            fg=WIN_TEXT,
            anchor="w",
        )
        self.status_label.pack(fill=tk.X)

        self.progress_bar = SegmentedProgressBar(prog_inner, segments=28, height=24)
        self.progress_bar.pack(fill=tk.X, pady=(8, 4))
        self.progress_bar.set_indeterminate(True)

        self.file_label = tk.Label(
            prog_inner,
            text="",
            font=("MS Sans Serif", 8),
            bg=GROUP_BG,
            fg=WIN_DISABLED,
            anchor="w",
        )
        self.file_label.pack(fill=tk.X)

        # Percent readout like old driver installers
        self.pct_label = tk.Label(
            prog_inner,
            text="0%",
            font=("MS Sans Serif", 8, "bold"),
            bg=GROUP_BG,
            fg=WIN_TEXT,
            anchor="e",
        )
        self.pct_label.pack(fill=tk.X)

        log_grp = _group_frame(self.content, "Status")
        log_grp.pack(fill=tk.BOTH, expand=True, pady=(0, 0), ipady=4, ipadx=6)

        log_frame = tk.Frame(log_grp, bg=GROUP_BG)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(12, 6))

        self.log_text = tk.Text(
            log_frame,
            height=6,
            font=("Courier New", 8),
            bg=LOG_BG,
            fg=LOG_FG,
            relief=tk.SUNKEN,
            bd=2,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        scroll = tk.Scrollbar(log_frame, command=self.log_text.yview, relief=tk.RAISED)
        self.log_text.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _show_finish(self, success: bool) -> None:
        self._page = "finish"
        self._clear_content()
        self.btn_back.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL, text="Close")
        self.btn_next.config(state=tk.NORMAL if success else tk.DISABLED, text="Finish")

        icon = tk.Label(
            self.content,
            text="✓" if success else "✕",
            font=("MS Sans Serif", 28, "bold"),
            bg=WIN_BG,
            fg="#008000" if success else "#800000",
        )
        icon.pack(pady=(16, 4))

        title = "Installation Complete" if success else "Installation Failed"
        tk.Label(
            self.content,
            text=title,
            font=("MS Sans Serif", 10, "bold"),
            bg=WIN_BG,
            fg=WIN_TEXT,
        ).pack()

        if success:
            msg = (
                f"Sonex has been installed to:\n{self.install_dir.get()}\n\n"
                "Click Finish to close Setup, or use the shortcut in the bin folder."
            )
            if self._result and self._result.warnings:
                msg += "\n\nNote: Some optional components were not found:\n"
                msg += "\n".join(f"  • {w}" for w in self._result.warnings[:4])
        else:
            err = self._result.error if self._result else "Unknown error"
            msg = f"Setup could not complete.\n\n{err}"

        tk.Label(
            self.content,
            text=msg,
            font=("MS Sans Serif", 8),
            bg=WIN_BG,
            fg=WIN_TEXT,
            justify=tk.CENTER,
        ).pack(pady=12, padx=20)

        if success and sys.platform.startswith("win"):
            self.launch_var = tk.BooleanVar(value=True)
            tk.Checkbutton(
                self.content,
                text="Launch Sonex when Setup closes",
                variable=self.launch_var,
                font=("MS Sans Serif", 8),
                bg=WIN_BG,
            ).pack()

    # -- UI thread helpers --------------------------------------------------

    def _ui(self, fn, *args) -> None:
        self.root.after(0, lambda: fn(*args))

    def _set_status(self, message: str) -> None:
        if hasattr(self, "status_label"):
            self.status_label.config(text=message)

    def _set_progress(self, percent: int, message: str | None) -> None:
        if hasattr(self, "progress_bar"):
            self.progress_bar.set_indeterminate(False)
            self.progress_bar.set_percent(percent)
        if hasattr(self, "pct_label"):
            self.pct_label.config(text=f"{percent}%")
        if message and hasattr(self, "file_label"):
            self.file_label.config(text=message)

    def _append_log(self, message: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # -- Navigation ---------------------------------------------------------

    def _browse_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.install_dir.get(), title="Select Install Folder")
        if path:
            self.install_dir.set(path)

    def _on_back(self) -> None:
        if self._page == "options":
            self._show_welcome()

    def _on_next(self) -> None:
        if self._page == "welcome":
            self._show_options()
        elif self._page == "options":
            self._start_install()
        elif self._page == "finish":
            self._on_finish()

    def _on_cancel(self) -> None:
        if self._page == "installing" and self._install_thread and self._install_thread.is_alive():
            if not messagebox.askyesno("Sonex Setup", "Setup is not complete. Cancel installation?"):
                return
        self.root.destroy()

    def _on_finish(self) -> None:
        if (
            self._result
            and self._result.success
            and sys.platform.startswith("win")
            and getattr(self, "launch_var", None)
            and self.launch_var.get()
        ):
            bat = self._result.install_dir / "bin" / "sonex.bat"
            if bat.exists():
                subprocess.Popen([str(bat)], shell=True, cwd=str(self._result.install_dir))
        self.root.destroy()

    def _start_install(self) -> None:
        self._show_installing()

        opts = InstallOptions(
            install_dir=Path(self.install_dir.get()),
            ref=self.ref.get(),
            with_mfa=self.with_mfa.get(),
        )
        reporter = GuiReporter(self)

        def worker() -> None:
            result = run_install(opts, reporter)
            self._result = result
            self._ui(self._install_done, result)

        self._install_thread = threading.Thread(target=worker, daemon=True)
        self._install_thread.start()

    def _install_done(self, result) -> None:
        self.progress_bar.set_indeterminate(False)
        self.progress_bar.set_percent(100 if result.success else 0)
        self._show_finish(result.success)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    # Hide console on Windows when not frozen (python.exe → pythonw behavior)
    if sys.platform.startswith("win") and not getattr(sys, "frozen", False):
        try:
            import ctypes
            ctypes.windll.kernel32.FreeConsole()  # type: ignore[attr-defined]
        except Exception:
            pass

    app = SonexSetupWizard()
    app.run()


if __name__ == "__main__":
    main()
