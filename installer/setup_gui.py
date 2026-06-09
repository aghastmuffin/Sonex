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
import tkinter.font as tkfont
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

WIN_W, WIN_H = 540, 420


def _enable_win_dpi() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:
            pass


def _pick_font(root: tk.Misc, size: int, bold: bool = False) -> tuple[str, int, str] | tuple[str, int]:
    if sys.platform.startswith("win"):
        families = ("Tahoma", "Segoe UI", "MS Sans Serif", "Arial")
    elif sys.platform == "darwin":
        families = ("Helvetica Neue", "Helvetica", "Lucida Grande", "Arial")
    else:
        families = ("DejaVu Sans", "Liberation Sans", "Arial")

    weight = "bold" if bold else "normal"
    for family in families:
        try:
            tkfont.Font(root=root, family=family, size=size, weight=weight)
            return (family, size, "bold") if bold else (family, size)
        except tk.TclError:
            continue
    return (families[-1], size, "bold") if bold else (families[-1], size)


FONT: tuple = ("Arial", 8)
FONT_BOLD: tuple = ("Arial", 8, "bold")
FONT_TITLE: tuple = ("Arial", 10, "bold")
FONT_BANNER: tuple = ("Arial", 14, "bold")
FONT_SUB: tuple = ("Arial", 9)
FONT_MONO: tuple = ("Courier New", 8)


def _init_fonts(root: tk.Misc) -> None:
    global FONT, FONT_BOLD, FONT_TITLE, FONT_BANNER, FONT_SUB, FONT_MONO
    FONT = _pick_font(root, 8)
    FONT_BOLD = _pick_font(root, 8, bold=True)
    FONT_TITLE = _pick_font(root, 10, bold=True)
    FONT_BANNER = _pick_font(root, 14, bold=True)
    FONT_SUB = _pick_font(root, 9)
    if sys.platform.startswith("win"):
        mono_families = ("Courier New", "Consolas", "Lucida Console")
    elif sys.platform == "darwin":
        mono_families = ("Menlo", "Monaco", "Courier New")
    else:
        mono_families = ("DejaVu Sans Mono", "Liberation Mono", "Courier New")
    mono_size = 8 if sys.platform.startswith("win") else 9
    for family in mono_families:
        try:
            tkfont.Font(root=root, family=family, size=mono_size)
            FONT_MONO = (family, mono_size)
            break
        except tk.TclError:
            continue
    else:
        FONT_MONO = (mono_families[-1], mono_size)


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
        self.after_idle(self._redraw)

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
    btn = tk.Button(
        master,
        text=text,
        command=command,
        width=width,
        font=FONT,
        bg=WIN_FACE,
        fg=WIN_TEXT,
        activebackground="#DFDFDF",
        activeforeground=WIN_TEXT,
        relief=tk.RAISED,
        bd=2,
        highlightthickness=0,
        cursor="hand2",
    )
    if default:
        btn.configure(font=FONT_BOLD)
    return btn


def _win95_entry(master, textvariable: tk.StringVar, width: int = 42) -> tk.Entry:
    return tk.Entry(
        master,
        textvariable=textvariable,
        width=width,
        font=FONT,
        bg=LOG_BG,
        fg=WIN_TEXT,
        relief=tk.SUNKEN,
        bd=2,
        highlightthickness=0,
    )


def _win95_check(master, text: str, variable: tk.BooleanVar) -> tk.Checkbutton:
    return tk.Checkbutton(
        master,
        text=text,
        variable=variable,
        font=FONT,
        bg=master.cget("bg"),
        fg=WIN_TEXT,
        activebackground=master.cget("bg"),
        activeforeground=WIN_TEXT,
        selectcolor=master.cget("bg"),
        highlightthickness=0,
    )


def _group_frame(master, title: str) -> tuple[tk.Frame, tk.Frame]:
    """Return (wrapper, body). Pack wrapper; place children inside body."""
    wrapper = tk.Frame(master, bg=WIN_BG)
    body = tk.Frame(wrapper, bg=GROUP_BG, relief=tk.GROOVE, bd=2)
    body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    tk.Label(
        wrapper,
        text=f" {title} ",
        font=FONT_BOLD,
        bg=WIN_BG,
        fg=WIN_TEXT,
    ).place(in_=body, x=12, y=0, anchor="nw")

    return wrapper, body


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
        _enable_win_dpi()
        self.root = tk.Tk()
        _init_fonts(self.root)
        self.root.title("Sonex Setup")
        self.root.configure(bg=WIN_BG)
        self.root.resizable(False, False)

        self.root.geometry(f"{WIN_W}x{WIN_H}")
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
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - WIN_W) // 2
        y = (sh - WIN_H) // 2
        self.root.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")

    def _build_chrome(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # Title banner (Win9x installer header strip)
        self.banner = tk.Frame(self.root, bg=BANNER_BG, height=48)
        self.banner.grid(row=0, column=0, sticky="ew")
        self.banner.grid_propagate(False)

        tk.Label(
            self.banner,
            text="Sonex",
            font=FONT_BANNER,
            bg=BANNER_BG,
            fg=TITLE_FG,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(14, 0), pady=8)

        tk.Label(
            self.banner,
            text="Setup Wizard",
            font=FONT_SUB,
            bg=BANNER_BG,
            fg="#B0C4FF",
            anchor="w",
        ).pack(side=tk.LEFT, padx=(10, 0), pady=12)

        # Main content area
        self.content = tk.Frame(self.root, bg=WIN_BG)
        self.content.grid(row=1, column=0, sticky="nsew", padx=12, pady=(10, 6))

        sep = tk.Frame(self.root, bg=WIN_BG, height=2, relief=tk.GROOVE, bd=1)
        sep.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))

        # Button bar (always at bottom)
        self.btnbar = tk.Frame(self.root, bg=WIN_BG)
        self.btnbar.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))

        self.btn_back = _win95_button(self.btnbar, "< Back", self._on_back, width=9)
        self.btn_back.pack(side=tk.LEFT)

        self.btn_next = _win95_button(self.btnbar, "Next >", self._on_next, width=9, default=True)
        self.btn_next.pack(side=tk.RIGHT, padx=(6, 0))

        self.btn_cancel = _win95_button(self.btnbar, "Cancel", self._on_cancel, width=9)
        self.btn_cancel.pack(side=tk.RIGHT)

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
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        # Fake CD-ROM / box art panel
        art = tk.Canvas(left, width=88, height=128, bg=WIN_BG, highlightthickness=0)
        art.pack()
        art.create_rectangle(4, 4, 84, 124, fill="#808080", outline="#404040")
        art.create_rectangle(8, 8, 80, 120, fill="#A0A0A0", outline="")
        art.create_text(44, 54, text="SONEX", fill="#000080", font=FONT_TITLE)
        art.create_text(44, 78, text="v0.4", fill=WIN_TEXT, font=FONT)

        right = tk.Frame(self.content, bg=WIN_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            right,
            text="Welcome to the Sonex Setup Wizard",
            font=FONT_BOLD,
            bg=WIN_BG,
            fg=WIN_TEXT,
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 10))

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
            font=FONT,
            bg=WIN_BG,
            fg=WIN_TEXT,
            justify=tk.LEFT,
            anchor="nw",
            wraplength=340,
        ).pack(fill=tk.BOTH, expand=True)

    def _show_options(self) -> None:
        self._page = "options"
        self._clear_content()
        self.btn_back.config(state=tk.NORMAL)
        self.btn_next.config(text="Install", state=tk.NORMAL)

        tk.Label(
            self.content,
            text="Select Installation Options",
            font=FONT_BOLD,
            bg=WIN_BG,
            fg=WIN_TEXT,
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 10))

        dest_wrap, dest_body = _group_frame(self.content, "Destination Folder")
        dest_wrap.pack(fill=tk.X, pady=(0, 12))

        row = tk.Frame(dest_body, bg=GROUP_BG)
        row.pack(fill=tk.X, padx=12, pady=(10, 10))

        tk.Label(row, text="Install Sonex to:", font=FONT, bg=GROUP_BG).pack(anchor="w")
        path_row = tk.Frame(row, bg=GROUP_BG)
        path_row.pack(fill=tk.X, pady=(6, 0))
        _win95_entry(path_row, self.install_dir, width=46).pack(side=tk.LEFT, fill=tk.X, expand=True)
        _win95_button(path_row, "Browse...", self._browse_dir, width=9).pack(side=tk.LEFT, padx=(8, 0))

        comp_wrap, comp_body = _group_frame(self.content, "Components")
        comp_wrap.pack(fill=tk.X)

        inner = tk.Frame(comp_body, bg=GROUP_BG)
        inner.pack(fill=tk.X, padx=12, pady=(10, 10))

        _win95_check(
            inner,
            "Install MFA alignment environment (requires conda)",
            self.with_mfa,
        ).pack(anchor="w")

        tk.Label(
            inner,
            text=f"Release channel: {self.ref.get()}  (from GitHub)",
            font=FONT,
            bg=GROUP_BG,
            fg=WIN_DISABLED,
        ).pack(anchor="w", pady=(8, 0))

    def _show_installing(self) -> None:
        self._page = "installing"
        self._clear_content()
        self.btn_back.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.DISABLED)

        tk.Label(
            self.content,
            text="Installing Sonex",
            font=FONT_BOLD,
            bg=WIN_BG,
            fg=WIN_TEXT,
            anchor="w",
        ).pack(fill=tk.X)

        tk.Label(
            self.content,
            text="Please wait while Setup installs Sonex on your computer.",
            font=FONT,
            bg=WIN_BG,
            fg=WIN_TEXT,
            anchor="w",
        ).pack(fill=tk.X, pady=(4, 10))

        prog_wrap, prog_body = _group_frame(self.content, "Setup Progress")
        prog_wrap.pack(fill=tk.X, pady=(0, 10))

        prog_inner = tk.Frame(prog_body, bg=GROUP_BG)
        prog_inner.pack(fill=tk.X, padx=12, pady=(10, 10))

        self.status_label = tk.Label(
            prog_inner,
            text="Preparing installation...",
            font=FONT,
            bg=GROUP_BG,
            fg=WIN_TEXT,
            anchor="w",
        )
        self.status_label.pack(fill=tk.X)

        self.progress_bar = SegmentedProgressBar(prog_inner, segments=28, height=26)
        self.progress_bar.pack(fill=tk.X, pady=(8, 6))
        self.progress_bar.set_indeterminate(True)

        pct_row = tk.Frame(prog_inner, bg=GROUP_BG)
        pct_row.pack(fill=tk.X)

        self.file_label = tk.Label(
            pct_row,
            text="",
            font=FONT,
            bg=GROUP_BG,
            fg=WIN_DISABLED,
            anchor="w",
        )
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.pct_label = tk.Label(
            pct_row,
            text="0%",
            font=FONT_BOLD,
            bg=GROUP_BG,
            fg=WIN_TEXT,
            anchor="e",
            width=5,
        )
        self.pct_label.pack(side=tk.RIGHT)

        log_wrap, log_body = _group_frame(self.content, "Status")
        log_wrap.pack(fill=tk.BOTH, expand=True)

        log_frame = tk.Frame(log_body, bg=GROUP_BG)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 10))

        scroll = tk.Scrollbar(log_frame, command=None, relief=tk.SUNKEN, bd=1)
        self.log_text = tk.Text(
            log_frame,
            height=7,
            font=FONT_MONO,
            bg=LOG_BG,
            fg=LOG_FG,
            relief=tk.SUNKEN,
            bd=2,
            state=tk.DISABLED,
            wrap=tk.WORD,
            highlightthickness=0,
            yscrollcommand=scroll.set,
        )
        scroll.config(command=self.log_text.yview)
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
            text="[OK]" if success else "[X]",
            font=FONT_BANNER,
            bg=WIN_BG,
            fg="#008000" if success else "#800000",
        )
        icon.pack(pady=(20, 6))

        title = "Installation Complete" if success else "Installation Failed"
        tk.Label(
            self.content,
            text=title,
            font=FONT_TITLE,
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
                msg += "\n".join(f"  - {w}" for w in self._result.warnings[:4])
        else:
            err = self._result.error if self._result else "Unknown error"
            msg = f"Setup could not complete.\n\n{err}"

        tk.Label(
            self.content,
            text=msg,
            font=FONT,
            bg=WIN_BG,
            fg=WIN_TEXT,
            justify=tk.CENTER,
            wraplength=460,
        ).pack(pady=14, padx=24)

        if success and sys.platform.startswith("win"):
            self.launch_var = tk.BooleanVar(value=True)
            _win95_check(self.content, "Launch Sonex when Setup closes", self.launch_var).pack()

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
