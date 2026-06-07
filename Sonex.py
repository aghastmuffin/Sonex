from __future__ import annotations
from tkinter import dialog
"""Primary Entry Point for Sonex. Does not contain logic. NOT STANDALONE"""
import os
import sys
import json 

from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtCore import Qt, QElapsedTimer, QTimer, QUrl
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QFileDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QCheckBox,
    QLabel,
    QMessageBox,
    QGroupBox,
)
from PyQt6.QtCore import QProcess, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
import sys, os, json, subprocess, threading
import ui._updates as upd



import ui.frase_core as core


def _app_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


_repo_root = _app_root()
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from backbone.ltra.languages import (
    DETECT_LANGUAGE,
    TRANSLATION_MODES,
    get_system_language_code,
    sorted_language_items,
)
from backbone.ltra.whisper_models import (
    WHISPER_MODEL_OPTIONS,
    download_whisper_model,
    is_whisper_model_installed,
    normalize_whisper_model_name,
    whisper_model_display_name,
)
#TODO: Swap out the loading bar in the gui for a splash screen with a progress bar and status text.
#settings variables
DEMUCS_MODEL = "htdemucs" #or "tasnet" or htdemucs 
DEMUCS_STEMS = "default" #or "other" or "both" or vocals
WHISPER_MODEL = "large-v3-turbo" #or "tiny", "base", "small", "large-v2" DEFAULT
WHISPER_BEAMSIZE = 5
WHISPER_PAT = 2
WHISPER_BESTOF = 3
GPU = False
UPDATES = True

# Set to a .ttf path for a custom app font, or None for system default.
UI_FONT_PATH = "ui/assets/Darker Grotesque.ttf" #TODO: Make OS agnostic

APP_STYLE = """
QWidget {
    background-color: #1e1e1e;
    color: #f5f5f5;
}
QLabel {
    background-color: transparent;
}
QGroupBox {
    background-color: transparent;
    border: 1px solid #4a4a4a;
    border-radius: 8px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #f5f5f5;
    background-color: transparent;
}
QDialog {
    background-color: #181818;
}
QPushButton {
    background-color: #3a3a3a;
    border: 1px solid #5a5a5a;
    border-radius: 8px;
    padding: 8px 14px;
}
QPushButton:hover {
    background-color: #4a4a4a;
}
QPushButton:disabled {
    background-color: #373737;
    color: #888888;
}
QPushButton#startBtn:enabled {
    background-color: #46825f;
    border-color: #5a9a75;
}
QPushButton#forceNativeBtn:checked {
    background-color: #5a7864;
    border-color: #6a9a7a;
}
QComboBox {
    background-color: #343434;
    border: 1px solid #5a5a5a;
    border-radius: 8px;
    padding: 6px 10px;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QProgressBar {
    background-color: #373737;
    border: none;
    border-radius: 4px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0,
        stop:0 #5a78be, stop:1 #9ec8ff);
    border-radius: 4px;
}
QLabel#lyricsLabel {
    font-size: 36px;
    line-height: 1.25;
}
QLabel#statusOk {
    color: #bedcbe;
}
QLabel#statusWarn {
    color: #e6aa78;
}
QLabel#statusErr {
    color: #e67878;
}
"""

_ui_font_family: str | None = None


def _resolve_ui_font_family() -> str | None:
    global _ui_font_family
    if not UI_FONT_PATH or not os.path.exists(UI_FONT_PATH):
        return None
    if _ui_font_family is None:
        fid = QFontDatabase.addApplicationFont(UI_FONT_PATH)
        if fid >= 0:
            families = QFontDatabase.applicationFontFamilies(fid)
            if families:
                _ui_font_family = families[0]
    print("font loaded successfully")
    return _ui_font_family


def load_ui_font(size: int, bold: bool = False) -> QFont:
    family = _resolve_ui_font_family()
    if family:
        font = QFont(family, size)
        font.setBold(bold)
        return font
    font = QFont()
    font.setPointSize(size)
    font.setBold(bold)
    return font


def apply_app_styles(app: QApplication | None = None):
    app = app or QApplication.instance()
    if app is None:
        return
    app.setStyleSheet(APP_STYLE)
    if _resolve_ui_font_family():
        app.setFont(load_ui_font(13))


def default_output_root():
    app_name = "Sonex"
    if sys.platform.startswith("darwin"):
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    elif sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, app_name, "outputs")


def resolve_output_root(output_root_arg=None):
    if output_root_arg:
        return output_root_arg
    env_root = os.environ.get("SONEX_OUTPUT_ROOT")
    if env_root:
        return env_root
    return default_output_root()


def _is_frozen():
    return bool(getattr(sys, "frozen", False))


def resolve_worker_script() -> str:
    return os.path.join(_app_root(), "ui", "_worker.py")


def resolve_worker_command(worker_script=None):
    if worker_script is None:
        worker_script = resolve_worker_script()
    if not _is_frozen():
        return sys.executable, [worker_script]

    override = os.environ.get("SONEX_WORKER_BIN")
    candidates = []
    if override:
        candidates.append(override)

    exe_dir = os.path.dirname(sys.executable)
    exe_suffix = ".exe" if sys.platform.startswith("win") else ""
    candidates.append(os.path.join(exe_dir, f"sonex-worker{exe_suffix}"))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate, []

    return None, []


def resolve_worker_cwd():
    if _is_frozen():
        return os.path.dirname(sys.executable)
    return _app_root()


def notify(title, message, parent=None):
    app = QApplication.instance()
    if app is None:
        print(f"{title}: {message}")
        return
    QMessageBox.information(parent, title, message)


def dialogue(title, message, parent=None):
    app = QApplication.instance()
    if app is None:
        print(f"{title}: {message}")
        return
    QMessageBox.warning(parent, title, message)


def ask_yes_no(title, message, parent=None) -> bool:
    app = QApplication.instance()
    if app is None:
        print(f"{title}: {message}")
        return False
    reply = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes


class WhisperDownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, model_id, parent=None):
        super().__init__(parent)
        self.model_id = model_id

    def run(self):
        try:
            download_whisper_model(
                self.model_id,
                progress_cb=lambda value, label: self.progress.emit(int(value), label),
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class WhisperInstallDialog(QDialog):
    def __init__(self, model_id, parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self._success = False

        self.setWindowTitle("Install Whisper")
        self.setModal(True)
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        self.status_label = QLabel(f"Downloading Whisper model: {model_id}")
        layout.addWidget(self.status_label)

        self.prog_bar = QProgressBar(self)
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setFormat("Starting download...")
        layout.addWidget(self.prog_bar)

        self.worker = WhisperDownloadWorker(model_id, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_success)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, value, label):
        self.prog_bar.setValue(max(0, min(100, int(value))))
        if label:
            self.prog_bar.setFormat(label)
            self.status_label.setText(label)
        QApplication.processEvents()

    def _on_success(self):
        self._success = True
        self.prog_bar.setValue(100)
        self.prog_bar.setFormat(f"Downloaded {self.model_id}")
        self.accept()

    def _on_failed(self, message):
        self.prog_bar.setFormat("Download failed")
        dialogue("Install Whisper", f"Failed to download {self.model_id}:\n{message}", parent=self)
        self.reject()

    @property
    def success(self):
        return self._success


def ensure_whisper_model_installed(model_id, parent=None) -> bool:
    if is_whisper_model_installed(model_id):
        return True
    if not ask_yes_no(
        "Install Whisper",
        f"The Whisper model '{model_id}' is not installed.\n\nDownload it now?",
        parent=parent,
    ):
        return False
    dialog = WhisperInstallDialog(model_id, parent=parent)
    dialog.exec()
    return dialog.success


def populate_whisper_model_combo(combo, selected_model_id):
    combo.clear()
    for model_id, tooltip in WHISPER_MODEL_OPTIONS:
        combo.addItem(whisper_model_display_name(model_id), model_id)
        idx = combo.count() - 1
        combo.setItemData(idx, tooltip, Qt.ItemDataRole.ToolTipRole)

    for idx in range(combo.count()):
        if combo.itemData(idx) == selected_model_id:
            combo.setCurrentIndex(idx)
            return
    combo.setCurrentIndex(0)


def refresh_whisper_model_combo_labels(combo):
    current_model_id = combo.currentData()
    for idx in range(combo.count()):
        model_id = combo.itemData(idx)
        combo.setItemText(idx, whisper_model_display_name(model_id))
    for idx in range(combo.count()):
        if combo.itemData(idx) == current_model_id:
            combo.setCurrentIndex(idx)
            return


class Notification(QDialog):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Notification")
        self.setModal(True)
        # Add your notification content here
        layout = QVBoxLayout(self)
        label = QLabel(message, self) 
        layout.addWidget(label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, self)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def _whisper_cuda_available() -> bool:
    try:
        import ctranslate2

        ctranslate2.get_supported_compute_types("cuda")
        return True
    except Exception:
        return False


def detect_gpu_acceleration() -> tuple[bool, str]:
    """Return whether an accelerator is available and a user-facing status label."""
    try:
        import torch

        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA device"
            if _whisper_cuda_available():
                return True, (
                    f"NVIDIA GPU available ({name}). "
                    "Demucs, Whisper, and translation can use CUDA."
                )
            return True, (
                f"NVIDIA GPU available ({name}). "
                "Demucs and translation use CUDA; Whisper runs on CPU."
            )

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return True, (
                "Apple Metal (MPS) GPU available. "
                "Demucs and translation use Metal; Whisper runs on CPU."
            )
    except Exception as exc:
        return False, f"GPU check failed: {exc}"

    return False, "No compatible GPU detected. All processing uses CPU."


class GPUCheckWorker(QObject):
    finished = pyqtSignal(bool, str)

    def run(self):
        available, message = detect_gpu_acceleration()
        self.finished.emit(available, message)

class AdvancedSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings") #TODO: Preflight check pretrained MFA models and check which will need to be used, maybe quick pre-analysis? 
        self.setModal(True)

        form = QFormLayout(self)

        self.demucs_model_input = QComboBox(self)
        self.demucs_model_input.addItems(["htdemucs", "htdemucs_ft", "mdx_extra"])
        self.demucs_model_input.setCurrentText(settings["demucs_model"])
        form.addRow("Demucs model", self.demucs_model_input)

        self.demucs_stems_input = QComboBox(self)
        self.demucs_stems_input.addItems(["default", "vocals", "other", "both"])
        self.demucs_stems_input.setCurrentText(settings["demucs_stems"])
        form.addRow("Demucs stems", self.demucs_stems_input)


        self.whisper_model_input = QComboBox(self)
        populate_whisper_model_combo(
            self.whisper_model_input,
            normalize_whisper_model_name(settings.get("whisper_model")),
        )
        form.addRow("Whisper model", self.whisper_model_input)

        self.whisper_beam_input = QSpinBox(self)
        self.whisper_beam_input.setRange(1, 20)
        self.whisper_beam_input.setValue(int(settings["whisper_beam_size"]))
        form.addRow("Whisper beam size", self.whisper_beam_input)

        self.whisper_pat_input = QSpinBox(self)
        self.whisper_pat_input.setRange(1, 10)
        self.whisper_pat_input.setValue(int(settings["whisper_patience"]))
        form.addRow("Whisper patience", self.whisper_pat_input)

        self.whisper_bestof_input = QSpinBox(self)
        self.whisper_bestof_input.setRange(1, 20)
        self.whisper_bestof_input.setValue(int(settings["whisper_best_of"]))
        form.addRow("Whisper best_of", self.whisper_bestof_input)

        """self.MFA_target_in = QComboBox(self)
        self.MFA_target_in.addItems(["default", "vocals", "other", "both"])
        self.MFA_target_in.setCurrentText(settings["MFA_target"])
        form.addRow("MFA - FromLang:", self.MFA_target_in)

        self.MFA_target_in = QComboBox(self)
        self.MFA_target_in.addItems(["default", "vocals", "other", "both"])
        self.MFA_target_in.setCurrentText(settings["MFA_target"])
        form.addRow("MFA - FromLang:", self.MFA_target_in)"""

        self._saved_gpu = bool(settings["gpu"])
        self.gpu_input = QCheckBox("Enable GPU acceleration")
        self.gpu_input.setChecked(self._saved_gpu)
        self.gpu_input.setEnabled(False)

        self.gpu_status = QLabel("Checking GPU...")
        self.gpu_status.setObjectName("statusWarn")
        self.gpu_status.setWordWrap(True)
        form.addRow(self.gpu_input, self.gpu_status)

        self._gpu_thread = QThread()
        self._gpu_worker = GPUCheckWorker()
        self._gpu_worker.moveToThread(self._gpu_thread)
        self._gpu_thread.started.connect(self._gpu_worker.run)
        self._gpu_worker.finished.connect(self._on_gpu_result)
        self._gpu_worker.finished.connect(self._gpu_worker.deleteLater)
        self._gpu_worker.finished.connect(self._gpu_thread.quit)
        self._gpu_thread.finished.connect(self._gpu_thread.deleteLater)
        self._gpu_thread.start()

        self.flatten_audio = QCheckBox("Flatten audio (pitch shift and bandpass, can help with alignment quality but is a lossy operation)")
        self.flatten_audio.setChecked(bool(settings.get("flatten", settings.get("flatten_audio", False))))

        self.phoneme_timestamps = QCheckBox("Phoneme-level timestamps (MFA alignment; richer karaoke highlighting)")
        self.phoneme_timestamps.setChecked(bool(settings.get("phoneme_timestamps", True)))

        form.addRow(self.flatten_audio)
        form.addRow(self.phoneme_timestamps)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_accept(self):
        model_id = self.whisper_model_input.currentData()
        if not ensure_whisper_model_installed(model_id, parent=self):
            return
        refresh_whisper_model_combo_labels(self.whisper_model_input)
        self.accept()

    def _set_gpu_status(self, message: str, *, ok: bool):
        self.gpu_status.setText(message)
        self.gpu_status.setObjectName("statusOk" if ok else "statusWarn")
        self.gpu_status.style().unpolish(self.gpu_status)
        self.gpu_status.style().polish(self.gpu_status)

    def _on_gpu_result(self, available: bool, message: str):
        if available:
            self.gpu_input.setEnabled(True)
            self.gpu_input.setChecked(self._saved_gpu)
            self._set_gpu_status(message, ok=True)
        else:
            self.gpu_input.setEnabled(False)
            self.gpu_input.setChecked(False)
            self._set_gpu_status(message, ok=False)

    def get_settings(self):
        return {
            "demucs_model": self.demucs_model_input.currentText(),
            "demucs_stems": self.demucs_stems_input.currentText(),
            "whisper_model": self.whisper_model_input.currentData(),
            "whisper_beam_size": int(self.whisper_beam_input.value()),
            "whisper_patience": int(self.whisper_pat_input.value()),
            "whisper_best_of": int(self.whisper_bestof_input.value()),
            "gpu": bool(self.gpu_input.isChecked()),
            "flatten": bool(self.flatten_audio.isChecked()),
            "phoneme_timestamps": bool(self.phoneme_timestamps.isChecked()),
        }

def resolve_app_icon_path():
    base_dir = os.path.dirname(__file__)
    candidate_paths = [
        os.path.join(base_dir, "ui", "assets", "sonex0high-res"),
        os.path.join(base_dir, "ui", "assets", "sonex0high-res.png"),
        os.path.join(base_dir, "ui", "assets", "sonex-high-resolution-logo.png"),
        os.path.join(base_dir, "ui", "assets", "sonex-high-resolution-logo-transparent.png"),
    ]
    for icon_path in candidate_paths:
        if os.path.exists(icon_path):
            return icon_path
    return None


class SplashWindow(QWidget):
    """Top-level splash screen with image backdrop and stage progress overlays."""

    def __init__(self, parent=None):
        super().__init__(None)
        self._parent_window = parent
        self.setWindowTitle("SONEX (taeson.co) - Processing")
        self.setWindowFlags(
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        base_dir = os.path.dirname(__file__)
        splash_path = os.path.join(base_dir, "ui", "assets", "sonex_splash.png")
        splash_pix = QPixmap(splash_path)
        if splash_pix.isNull():
            splash_pix = QPixmap(920, 520)
            splash_pix.fill(Qt.GlobalColor.black)

        # Scale to roughly 1/8 the original area for a compact splash.
        scale_factor = 0.35
        target_width = max(280, int(splash_pix.width() * scale_factor))
        target_height = max(170, int(splash_pix.height() * scale_factor))
        splash_pix = splash_pix.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.setFixedSize(splash_pix.size())

        bg_label = QLabel(self)
        bg_label.setPixmap(splash_pix)
        bg_label.setScaledContents(True)
        bg_label.setGeometry(0, 0, self.width(), self.height())

        margin = max(10, int(self.width() * 0.03))
        overlay_height = max(88, int(self.height() * 0.44))
        overlay = QWidget(self)
        overlay.setGeometry(margin, self.height() - overlay_height - margin, self.width() - (margin * 2), overlay_height)
        overlay.setStyleSheet("background-color: rgba(0, 0, 0, 145); border-radius: 12px;")

        layout = QVBoxLayout(overlay)
        layout.setSpacing(4)
        layout.setContentsMargins(10, 8, 10, 8)

        title = QLabel(f"Sonex is busy processing your request")
        title.setStyleSheet("font-size: 13px; font-weight: 700; color: #ffffff;")
        layout.addWidget(title)

        overall_label = QLabel("Overall")
        overall_label.setStyleSheet("font-size: 10px; color: #d7d7d7;")
        layout.addWidget(overall_label)
        self.prog_bar = QProgressBar(overlay)
        self.prog_bar.setValue(0)
        self.prog_bar.setFormat("Idle")
        layout.addWidget(self.prog_bar)

        demucs_label = QLabel("Demucs")
        demucs_label.setStyleSheet("font-size: 10px; color: #d7d7d7;")
        layout.addWidget(demucs_label)
        self.demucs_prog_bar = QProgressBar(overlay)
        self.demucs_prog_bar.setValue(0)
        self.demucs_prog_bar.setFormat("Demucs idle")
        self.demucs_prog_bar.setVisible(False)
        layout.addWidget(self.demucs_prog_bar)

        whisper_label = QLabel("Whisper")
        whisper_label.setStyleSheet("font-size: 10px; color: #d7d7d7;")
        layout.addWidget(whisper_label)
        self.whisper_prog_bar = QProgressBar(overlay)
        self.whisper_prog_bar.setValue(0)
        self.whisper_prog_bar.setFormat("Whisper idle")
        self.whisper_prog_bar.setVisible(False)
        layout.addWidget(self.whisper_prog_bar)

        bar_style = (
            "QProgressBar {"
            "  background-color: rgba(255,255,255,0.10);"
            "  color: #ffffff;"
            "  border: 1px solid rgba(255,255,255,0.30);"
            "  border-radius: 5px;"
            "  text-align: center;"
            "  height: 12px;"
            "}"
            "QProgressBar::chunk {"
            "  background-color: #47a8ff;"
            "  border-radius: 4px;"
            "}"
        )
        self.prog_bar.setStyleSheet(bar_style)
        self.demucs_prog_bar.setStyleSheet(bar_style)
        self.whisper_prog_bar.setStyleSheet(bar_style)

        self._center_on_screen()

    def _center_on_screen(self):
        target = self._parent_window if self._parent_window is not None else self
        screen = target.screen() if target is not None else QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + (geo.height() - self.height()) // 2
        self.move(x, y)

    def set_progress(self, value, label=None):
        self.prog_bar.setValue(max(0, min(100, int(value))))
        if label is not None:
            self.prog_bar.setFormat(label)
        QApplication.processEvents()

    def set_demucs_active(self, active):
        self.demucs_prog_bar.setVisible(bool(active))
        if active:
            self.demucs_prog_bar.setValue(0)
            self.demucs_prog_bar.setFormat("Demucs separating stems...")
        QApplication.processEvents()

    def set_demucs_progress(self, value, label=None):
        self.demucs_prog_bar.setValue(max(0, min(100, int(value))))
        if label is not None:
            self.demucs_prog_bar.setFormat(label)
        QApplication.processEvents()

    def set_whisper_active(self, active):
        self.whisper_prog_bar.setVisible(bool(active))
        if active:
            self.whisper_prog_bar.setValue(0)
            self.whisper_prog_bar.setFormat("Whisper transcribing...")
        QApplication.processEvents()

    def set_whisper_progress(self, value, label=None):
        self.whisper_prog_bar.setValue(max(0, min(100, int(value))))
        if label is not None:
            self.whisper_prog_bar.setFormat(label)
        QApplication.processEvents()

    def finish(self):
        """Called when processing is complete"""
        self.set_demucs_active(False)
        self.set_whisper_active(False)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.advanced_settings = {
            "demucs_model": DEMUCS_MODEL,
            "demucs_stems": DEMUCS_STEMS,
            "whisper_model": WHISPER_MODEL,
            "whisper_beam_size": WHISPER_BEAMSIZE,
            "whisper_patience": WHISPER_PAT,
            "whisper_best_of": WHISPER_BESTOF,
            "gpu": GPU,
            "flatten": True,
            "phoneme_timestamps": True,
        }

        self.setWindowTitle("SONEX - Analyzer")
        self.setGeometry(100, 100, 460, 420)

        layout = QFormLayout()
        self.system_language = get_system_language_code()

        translation_group = QGroupBox("Basic Configuration")
        translation_layout = QFormLayout(translation_group)

        self.lang_input = QComboBox(self)
        for name, code in sorted_language_items(include_detect=True):
            self.lang_input.addItem(name, code)
        self.lang_input.setCurrentIndex(0)
        translation_layout.addRow("From:", self.lang_input)

        self.lang_to = QComboBox(self)
        for name, code in sorted_language_items(include_detect=False):
            self.lang_to.addItem(name, code)
        self._set_combo_by_data(self.lang_to, self.system_language)
        translation_layout.addRow("To:", self.lang_to)

        self.translation_mode_input = QComboBox(self)
        for label, mode in TRANSLATION_MODES:
            self.translation_mode_input.addItem(label, mode)
        self.translation_mode_input.currentIndexChanged.connect(self._sync_translation_controls)
        translation_layout.addRow("Translation:", self.translation_mode_input)

        self.translation_hint = QLabel(
            "Leave From on Detect language for Whisper auto-detect. "
            "To defaults to your system language. Whisper translation always outputs English."
        )
        self.translation_hint.setWordWrap(True)
        self.translation_hint.setStyleSheet("background-color: transparent; color: #666666; font-size: 11px;")
        translation_layout.addRow(self.translation_hint)
        layout.addRow(translation_group)

        self.filebtn = QPushButton("Choose Media File (.MP3 Only)")
        self.filebtn.clicked.connect(self.on_want_file)
        layout.addRow(self.filebtn)
        
        self.button_condenser = QHBoxLayout()

        self.viewer_launch = QPushButton("Open Lyrics Viewer")
        self.viewer_launch.clicked.connect(self.open_viewer)
        self.button_condenser.addWidget(self.viewer_launch)

        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.open_advanced_settings)
        self.button_condenser.addWidget(self.advanced_button)
        layout.addRow(self.button_condenser)

        self.button = QPushButton("Choose File First")
        self.button.setEnabled(False)
        self.button.clicked.connect(self.on_button_click)
        layout.addRow(self.button)
        
        self.prog_bar = QProgressBar(self)
        self.prog_bar.setGeometry(50, 100, 250, 30)
        self.prog_bar.setValue(0)
        self.prog_bar.setFormat("Idle")

        self.demucs_prog_bar = QProgressBar(self)
        self.demucs_prog_bar.setGeometry(50, 100, 250, 30)
        self.demucs_prog_bar.setValue(0)
        self.demucs_prog_bar.setFormat("Demucs idle")
        self.demucs_prog_bar.setVisible(False)

        self.whisper_prog_bar = QProgressBar(self)
        self.whisper_prog_bar.setGeometry(50, 100, 250, 30)
        self.whisper_prog_bar.setValue(0)
        self.whisper_prog_bar.setFormat("Whisper idle")
        self.whisper_prog_bar.setVisible(False)

        self.pipeline_process = None
        self.splash_window = None
        self.audiobase = None

        layout.addRow(self.prog_bar)
        layout.addRow(self.demucs_prog_bar)
        layout.addRow(self.whisper_prog_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self._sync_translation_controls()

    @staticmethod
    def _set_combo_by_data(combo, data):
        for idx in range(combo.count()):
            if combo.itemData(idx) == data:
                combo.setCurrentIndex(idx)
                return

    def _sync_translation_controls(self):
        mode = self.translation_mode_input.currentData() or "none"
        translating = mode != "none"
        #self.lang_to.setEnabled(translating)
        if translating:
            if self.lang_to.currentData() is None:
                self._set_combo_by_data(self.lang_to, self.system_language)
            self.translation_hint.setText(
                "From defaults to Detect language (Whisper auto-detect). "
                f"To defaults to your system language ({self.system_language.upper()}). "
                "Whisper mode translates to English only; OpusMT/NLLB uses the To language."
            )
        else:
            self.translation_hint.setText(
                "Leave From on Detect language for Whisper auto-detect. "
                "Enable a translation mode to translate into your system language by default."
            )

    def on_want_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select an audio file", "", "Audio Files (*.mp3)")
        self.filebtn.setText(os.path.basename(self.file_path) if self.file_path else "Choose Media File (.MP3 Only)")
        self.button.setText("Listo" if self.file_path else "Choose File First")
        self.button.setEnabled(bool(self.file_path))

    def set_progress(self, value, label=None):
        self.prog_bar.setValue(max(0, min(100, int(value))))
        if label is not None:
            self.prog_bar.setFormat(label)
        QApplication.processEvents()

    def set_demucs_active(self, active):
        self.demucs_prog_bar.setVisible(bool(active))
        if active:
            self.demucs_prog_bar.setValue(0)
            self.demucs_prog_bar.setFormat("Demucs separating stems...")
        QApplication.processEvents()

    def set_demucs_progress(self, value, label=None):
        self.demucs_prog_bar.setValue(max(0, min(100, int(value))))
        if label is not None:
            self.demucs_prog_bar.setFormat(label)
        QApplication.processEvents()

    def set_whisper_active(self, active):
        self.whisper_prog_bar.setVisible(bool(active))
        if active:
            self.whisper_prog_bar.setValue(0)
            self.whisper_prog_bar.setFormat("Whisper transcribing...")
        QApplication.processEvents()

    def set_whisper_progress(self, value, label=None):
        self.whisper_prog_bar.setValue(max(0, min(100, int(value))))
        if label is not None:
            self.whisper_prog_bar.setFormat(label)
        QApplication.processEvents()

    def on_button_click(self):
        from_lang = self.lang_input.currentData()
        self.lang_code = from_lang if from_lang not in (None, DETECT_LANGUAGE) else None
        self.translation_mode = self.translation_mode_input.currentData() or "none"
        if not getattr(self, "file_path", None):
            return

        if self.pipeline_process is not None and self.pipeline_process.state() != QProcess.ProcessState.NotRunning:
            return

        whisper_model_id = normalize_whisper_model_name(self.advanced_settings.get("whisper_model"))
        if not ensure_whisper_model_installed(whisper_model_id, parent=self):
            return

        # Create and show splash window
        self.splash_window = SplashWindow(self)
        self.splash_window.show()
        self.splash_window.raise_()
        self.splash_window.activateWindow()
        self.splash_window.set_progress(5, "Starting...")

        self.button.setEnabled(False)
        self.set_progress(5, "Starting...")
        self.set_demucs_active(False)
        self.set_whisper_active(False)

        program, base_args = resolve_worker_command()
        if not program:
            if self.splash_window:
                self.splash_window.finish()
                self.splash_window.close()
                self.splash_window = None
            self.button.setEnabled(True)
            dialogue(
                "Missing worker binary",
                "Unable to locate the Sonex worker binary. Set SONEX_WORKER_BIN or place 'sonex-worker' next to the app executable.",
                parent=self,
            )
            return

        project_root = resolve_worker_cwd()
        output_root = resolve_output_root()
        os.makedirs(output_root, exist_ok=True)
        lang_arg = DETECT_LANGUAGE if self.lang_code is None else self.lang_code
        target_lang_arg = self.lang_to.currentData() or self.system_language
        translation_mode_arg = self.translation_mode if self.translation_mode else "none"
        settings_arg = json.dumps(self.advanced_settings)

        self.pipeline_process = QProcess(self)
        self.pipeline_process.setWorkingDirectory(project_root)
        self.pipeline_process.setProgram(program)
        self.pipeline_process.setArguments(base_args + [
            self.file_path,
            lang_arg,
            translation_mode_arg,
            settings_arg,
            target_lang_arg,
            output_root,
        ])
        self.pipeline_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.pipeline_process.readyReadStandardOutput.connect(self.on_pipeline_stdout)
        self.pipeline_process.finished.connect(self.on_pipeline_finished)
        self.pipeline_process.errorOccurred.connect(self.on_pipeline_process_error)
        self.pipeline_process.start()

    def open_advanced_settings(self):
        self.advanced_button.setText("Loading Advanced Settings...")
        import time
        time.sleep(0.04) #XXX: So janky
        self.advanced_button.repaint()              
        QApplication.processEvents()    
        def private_open():
            dialog = AdvancedSettingsDialog(self.advanced_settings, self)
            if dialog.exec():
                self.advanced_settings = dialog.get_settings()
            self.advanced_button.setText("Advanced Settings")
        QTimer.singleShot(0, lambda: private_open())      

        

    def open_viewer(self):
        generarfrase(parent=self)

    def on_pipeline_stdout(self):
        if self.pipeline_process is None:
            return
        output = bytes(self.pipeline_process.readAllStandardOutput()).decode(errors="replace")
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("PROGRESS|"):
                parts = line.split("|", 2)
                if len(parts) == 3:
                    try:
                        self.set_progress(int(parts[1]), parts[2])
                        if self.splash_window:
                            self.splash_window.set_progress(int(parts[1]), parts[2])
                    except ValueError:
                        pass
                continue
            if line.startswith("AUDIOBASE|"):
                parts = line.split("|", 1)
                if len(parts) == 2 and parts[1]:
                    self.audiobase = parts[1]
                continue
            if line.startswith("DEMUCS_ACTIVE|"):
                parts = line.split("|", 1)
                if len(parts) == 2:
                    self.set_demucs_active(parts[1] == "1")
                    if self.splash_window:
                        self.splash_window.set_demucs_active(parts[1] == "1")
                continue
            if line.startswith("DEMUCS_PROGRESS|"):
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    try:
                        progress_value = int(parts[1])
                        label = parts[2] if len(parts) == 3 else None
                        self.set_demucs_progress(progress_value, label)
                        if self.splash_window:
                            self.splash_window.set_demucs_progress(progress_value, label)
                    except ValueError:
                        pass
                continue
            if line.startswith("WHISPER_ACTIVE|"):
                parts = line.split("|", 1)
                if len(parts) == 2:
                    self.set_whisper_active(parts[1] == "1")
                    if self.splash_window:
                        self.splash_window.set_whisper_active(parts[1] == "1")
                continue
            if line.startswith("WHISPER_PROGRESS|"):
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    try:
                        progress_value = int(parts[1])
                        label = parts[2] if len(parts) == 3 else None
                        self.set_whisper_progress(progress_value, label)
                        if self.splash_window:
                            self.splash_window.set_whisper_progress(progress_value, label)
                    except ValueError:
                        pass
                continue
            if line.startswith("LANG|"):
                parts = line.split("|", 1)
                if len(parts) == 2 and parts[1]:
                    self.lang_code = parts[1]
                continue
            print(line)

    def on_pipeline_finished(self, exit_code, exit_status):
        if exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit:
            self.set_progress(100, "Done")
            print("Done")
        else:
            self.set_progress(0, f"Error ({exit_code})")
        self.set_demucs_active(False)
        self.set_whisper_active(False)
        
        # Close splash window if it exists
        if self.splash_window:
            self.splash_window.finish()
            self.splash_window.close()
            self.splash_window = None
        
        self.button.setEnabled(True)
        notify(
            "Processing Complete",
            f"Your file has been processed. Output directory: {self.audiobase}" if self.audiobase else "Your file has been processed.",
            parent=self,
        )
        if self.pipeline_process is not None:
            self.pipeline_process.deleteLater()
            self.pipeline_process = None

    def on_pipeline_process_error(self, _error):
        print("Processing Error: worker process failed")
        self.set_progress(0, "Error")
        self.set_demucs_active(False)
        self.set_whisper_active(False)
        
        # Close splash window if it exists
        if self.splash_window:
            self.splash_window.finish()
            self.splash_window.close()
            self.splash_window = None
        
        self.button.setEnabled(True)
        if self.pipeline_process is not None:
            self.pipeline_process.deleteLater()
            self.pipeline_process = None

"""lyrics viewer UI"""
class LoopProgressWidget(QWidget):
    """Minimal arc widget — no native circular progress in Qt widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self.setFixedSize(26, 26)

    def set_progress(self, progress: float | None):
        self._progress = 0.0 if progress is None else max(0.0, min(1.0, float(progress)))
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() / 2, self.height() / 2
        radius = 10
        thickness = 3

        base_pen = QPen(QColor(90, 90, 90), thickness)
        painter.setPen(base_pen)
        painter.drawEllipse(int(cx - radius), int(cy - radius), radius * 2, radius * 2)

        if self._progress <= 0:
            return

        span = int(360 * 16 * self._progress)
        arc_pen = QPen(QColor(120, 255, 170), thickness)
        painter.setPen(arc_pen)
        rect = int(cx - radius), int(cy - radius), radius * 2, radius * 2
        painter.drawArc(*rect, 90 * 16, -span)


class NoteStrengthPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._bars: list[QProgressBar] = []
        self._labels: list[QLabel] = []

        for i, note in enumerate(core.pitch_classes):
            bar = QProgressBar()
            bar.setOrientation(Qt.Orientation.Vertical)
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(90)
            label = QLabel(note)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: transparent; color: #dcdcdc; font-size: 11px;")
            self._bars.append(bar)
            self._labels.append(label)
            layout.addWidget(bar, 0, i)
            layout.addWidget(label, 1, i)

        self._na_label = QLabel("Note strength: [n/a]")
        self._na_label.setStyleSheet("background-color: transparent; color: #bebebe;")
        layout.addWidget(self._na_label, 0, 0, 2, len(core.pitch_classes))
        self._na_label.hide()

    def set_levels(self, levels):
        if levels is None:
            for bar in self._bars:
                bar.hide()
            for label in self._labels:
                label.hide()
            self._na_label.show()
            return

        self._na_label.hide()
        for bar, label, level in zip(self._bars, self._labels, levels):
            bar.show()
            label.show()
            bar.setValue(int(max(0.0, min(1.0, float(level))) * 100))


class FolderSelectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select generated folder")
        self.setMinimumWidth(760)

        self._eligible_dirs: list[dict] = []
        self._folder_path: str | None = None
        self._native_file: str | None = None
        self._translated_file: str | None = None
        self._resolved_folder: str | None = None
        self._force_native = False

        title = QLabel("Select generated folder")
        title.setFont(Viewer.load_app_font(28))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scan_label = QLabel("Auto-scan eligible folders:")
        self.combo = QComboBox()
        self.combo.setMinimumHeight(40)

        self.refresh_btn = QPushButton("Refresh")
        self.choose_btn = QPushButton("Choose Folder")
        self.force_native_btn = QPushButton("Force Native: OFF")
        self.force_native_btn.setObjectName("forceNativeBtn")
        self.force_native_btn.setCheckable(True)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setEnabled(False)

        row1 = QHBoxLayout()
        row1.addWidget(self.combo, stretch=1)
        row1.addWidget(self.refresh_btn)

        row2 = QHBoxLayout()
        row2.addWidget(self.choose_btn)
        row2.addWidget(self.force_native_btn)
        row2.addStretch()

        self.found_label = QLabel()
        self.folder_label = QLabel()
        self.resolved_label = QLabel()
        self.native_label = QLabel()
        self.translated_label = QLabel()
        self.mode_label = QLabel()
        self.hint_label = QLabel("Use dropdown or manual choose, then click Start")
        self.hint_label.setStyleSheet("background-color: transparent; color: #a5a5a5;")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for lbl in (
            self.found_label,
            self.folder_label,
            self.resolved_label,
            self.native_label,
            self.translated_label,
            self.mode_label,
        ):
            lbl.setWordWrap(True)

        buttons = QDialogButtonBox()
        buttons.addButton(self.start_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        cancel = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(scan_label)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.found_label)
        layout.addWidget(self.folder_label)
        layout.addWidget(self.resolved_label)
        layout.addWidget(self.native_label)
        layout.addWidget(self.translated_label)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.hint_label)
        layout.addWidget(buttons)

        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        self.refresh_btn.clicked.connect(self._refresh_dirs)
        self.choose_btn.clicked.connect(self._choose_folder)
        self.force_native_btn.toggled.connect(self._toggle_force_native)
        self.start_btn.clicked.connect(self._try_accept)
        cancel.clicked.connect(self.reject)

        self._refresh_dirs()

    def _apply_selection(self, index: int):
        if not (0 <= index < len(self._eligible_dirs)):
            return
        selection = self._eligible_dirs[index]
        self._folder_path = selection["folder"]
        self._native_file = selection.get("native")
        self._translated_file = selection.get("translated")
        self._resolved_folder = selection["folder"]
        self._update_status()

    def _on_combo_changed(self, index: int):
        self._apply_selection(index)

    def _refresh_dirs(self):
        self._eligible_dirs = core.refresh_eligible_dirs()
        self.combo.blockSignals(True)
        self.combo.clear()
        for item in self._eligible_dirs:
            self.combo.addItem(core.shorten_path(item["label"], max_len=78), item)
        self.combo.blockSignals(False)
        if self._eligible_dirs:
            self.combo.setCurrentIndex(0)
            self._apply_selection(0)
        else:
            self._native_file = None
            self._translated_file = None
            self._resolved_folder = None
            self._update_status()

    def _choose_folder(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose generated folder (or parent folder)",
            core.OUTPUT_ROOT or os.path.expanduser("~"),
        )
        if not selected:
            return
        self._folder_path = selected
        self._native_file, self._translated_file, self._resolved_folder = core._find_transcript_files(
            selected
        )
        self._update_status()

    def _toggle_force_native(self, checked: bool):
        self._force_native = checked
        self.force_native_btn.setText(f"Force Native: {'ON' if checked else 'OFF'}")
        self._update_status()

    def _can_start(self) -> bool:
        if self._force_native:
            return self._native_file is not None
        return (self._native_file is not None) or (self._translated_file is not None)

    def _update_status(self):
        self.found_label.setText(f"Found: {len(self._eligible_dirs)} eligible folders")
        self.folder_label.setText(f"Folder: {core.shorten_path(self._folder_path)}")
        self.resolved_label.setText(f"Resolved: {core.shorten_path(self._resolved_folder)}")

        native_color = "#bedcbe" if self._native_file else "#d2d2d2"
        trans_color = "#bedcbe" if self._translated_file else "#d2d2d2"
        self.native_label.setText(f"Native: {core.shorten_path(self._native_file)}")
        self.translated_label.setText(
            f"System/Translated: {core.shorten_path(self._translated_file)}"
        )
        self.native_label.setStyleSheet(f"background-color: transparent; color: {native_color};")
        self.translated_label.setStyleSheet(
            f"background-color: transparent; color: {trans_color};"
        )

        mode_text = ""
        mode_color = "#d2d2d2"
        if self._native_file and self._translated_file and not self._force_native:
            mode_text = "Mode: Dual transcript (native + system language)"
            mode_color = "#aadca8"
        elif self._force_native and self._native_file:
            mode_text = "Mode: Native only (forced)"
            mode_color = "#dcc9a0"
        elif self._native_file or self._translated_file:
            mode_text = "Mode: Single transcript"
            mode_color = "#dcc9a0"
        elif self._folder_path and not (self._native_file or self._translated_file):
            mode_text = "Could not find a transcript file in that folder tree."
            mode_color = "#e67878"
        elif self._force_native and not self._native_file and self._translated_file:
            mode_text = "Force native is ON but no native transcript was found."
            mode_color = "#e6aa78"
        elif not self._eligible_dirs:
            mode_text = "Auto-scan found none. Use Choose Folder for manual selection."
            mode_color = "#e6aa78"

        self.mode_label.setText(mode_text)
        self.mode_label.setStyleSheet(f"background-color: transparent; color: {mode_color};")
        self.start_btn.setEnabled(self._can_start())

    def _try_accept(self):
        if self._can_start():
            self.accept()

    def selection(self):
        return self._native_file, self._translated_file, self._resolved_folder, self._force_native


class Viewer(QMainWindow):
    WINDOW_TITLE = "🗣️ SONEX Lyrics, a TAESON.CO project."
    BRAND_DIR = os.path.join(os.path.dirname(__file__), "assets")

    def __init__(self, session: core.LyricsSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setWindowIcon(self.load_window_icon())
        self.resize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        self.header_label = QLabel(session.header_text)
        self.header_label.setStyleSheet("background-color: transparent; color: #ffffff; font-size: 13px;")

        self.orig_lyrics = QLabel()
        self.orig_lyrics.setObjectName("lyricsLabel")
        self.orig_lyrics.setTextFormat(Qt.TextFormat.RichText)
        self.orig_lyrics.setWordWrap(True)
        self.orig_lyrics.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.orig_lyrics.setFont(self.load_app_font(36))

        self.trans_lyrics = QLabel()
        self.trans_lyrics.setObjectName("lyricsLabel")
        self.trans_lyrics.setTextFormat(Qt.TextFormat.RichText)
        self.trans_lyrics.setWordWrap(True)
        self.trans_lyrics.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.trans_lyrics.setFont(self.load_app_font(36))
        self.trans_lyrics.hide()

        self.note_panel = NoteStrengthPanel()

        beat_row = QHBoxLayout()
        self.bpm_label = QLabel("BPM:   n/a |           ")
        self.bpm_label.setStyleSheet("background-color: transparent; color: #c8c8c8; font-size: 13px;")
        beat_row.addWidget(self.bpm_label)
        beat_row.addStretch()

        loop_row = QHBoxLayout()
        self.loop_ring = LoopProgressWidget()
        self.loop_label = QLabel("")
        self.loop_label.setStyleSheet("background-color: transparent; color: #bef5be; font-size: 13px;")
        loop_row.addWidget(self.loop_ring)
        loop_row.addWidget(self.loop_label)
        loop_row.addStretch()
        self._loop_row_widget = QWidget()
        self._loop_row_widget.setLayout(loop_row)
        self._loop_row_widget.hide()

        footer = QHBoxLayout()
        self.credits_label = QLabel("With <3 from Berkeley, Calif.")
        self.credits_label.setStyleSheet("background-color: transparent; color: #bebebe; font-size: 12px;")
        self.time_label = QLabel("Elapsed: 0 ms")
        self.time_label.setStyleSheet("background-color: transparent; color: #ffffff; font-size: 12px;")
        footer.addWidget(self.credits_label)
        footer.addStretch()
        footer.addWidget(self.time_label)

        root.addWidget(self.header_label)
        root.addWidget(self.orig_lyrics, stretch=2)
        root.addWidget(self.trans_lyrics, stretch=2)
        root.addWidget(self.note_panel)
        root.addLayout(beat_row)
        root.addWidget(self._loop_row_widget)
        root.addLayout(footer)

        if session.segments1:
            self.trans_lyrics.show()

        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        if session.audio_path:
            self._player.setSource(QUrl.fromLocalFile(session.audio_path))
            self._player.play()

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._frame_timer = QElapsedTimer()
        self._frame_timer.start()
        self._timer.start()

    @classmethod
    def apply_styles(cls, app: QApplication | None = None):
        apply_app_styles(app)

    @classmethod
    def load_app_font(cls, size: int, bold: bool = False) -> QFont:
        return load_ui_font(size, bold=bold)

    @classmethod
    def load_window_icon(cls) -> QIcon:
        candidates = [
            "resolution-logo.png",
            "sonex-high-resolution-logo.png",
            "sonex-high-resolution-logo-transparent.png",
            "sonex-high-resolution-logo-grayscale.png",
            "sonex-high-resolution-logo-grayscale-transparent.png",
        ]
        for name in candidates:
            path = os.path.join(cls.BRAND_DIR, name)
            if os.path.exists(path):
                icon = QIcon(path)
                if not icon.isNull():
                    return icon
        return QIcon()

    @classmethod
    def open(cls, parent=None) -> Viewer | None:
        cls.apply_styles()

        dialog = FolderSelectDialog(parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        native, translated, folder, force_native = dialog.selection()
        if not (native or translated):
            return None

        session = core.LyricsSession.load(native, translated, folder, force_native)
        if not session.audio_path:
            QMessageBox.warning(
                parent,
                "Audio not found",
                "Could not find a matching .mp3 next to the transcript folder.",
            )

        window = cls(session, parent=parent)
        window.show()
        window.raise_()
        window.activateWindow()

        print("job checking for MFA/fasterwhisper alignment accuracy")
        print("cleared for levi brown")
        return window

    def _elapsed_ms(self) -> int:
        pos = self._player.position()
        if pos >= 0:
            return int(pos)
        return 0

    def _tick(self):
        elapsed_ms = self._elapsed_ms()
        dt_ms = max(1, self._frame_timer.restart())

        orig_html = self.session.segment_html_at(elapsed_ms, track="orig")
        self.orig_lyrics.setText(
            f'<span style="color:#ffffff">{orig_html}</span>' if orig_html else ""
        )

        if self.session.segments1:
            trans_html = self.session.segment_html_at(elapsed_ms, track="trans")
            self.trans_lyrics.setText(
                f'<span style="color:#ffffff">{trans_html}</span>' if trans_html else ""
            )

        levels = self.session.update_note_bars(elapsed_ms, dt_ms)
        self.note_panel.set_levels(levels)

        bpm_text, beat_label, beat_visible = self.session.beat_display_at(elapsed_ms)
        color = "#ffdc50" if beat_visible else "#c8c8c8"
        self.bpm_label.setText(f"BPM: {bpm_text} | {beat_label}")
        self.bpm_label.setStyleSheet(f"background-color: transparent; color: {color}; font-size: 13px;")

        cycle = self.session.loop_display_at(elapsed_ms)
        if cycle and cycle.get("motif_text"):
            self._loop_row_widget.show()
            self.loop_ring.set_progress(cycle.get("progress"))
            self.loop_label.setText(f"Repeat~ [{cycle['motif_text']}]")
        else:
            self._loop_row_widget.hide()
            self.loop_ring.set_progress(0.0)
            self.loop_label.setText("")

        self.time_label.setText(f"Elapsed: {elapsed_ms} ms")

    def closeEvent(self, event):
        self._timer.stop()
        self._player.stop()
        super().closeEvent(event)

def generarfrase(parent=None) -> Viewer | None:
    return Viewer.open(parent)

def create_viewer():
    app = QApplication.instance()
    if app is None:
        #if called with cli args
        app = QApplication(sys.argv)
        Viewer.apply_styles(app)
        if Viewer.open() is None:
            return 0
        return app.exec()  # only when standalone
    else:
        # already inside a running Qt app
        Viewer.apply_styles(app)
        return Viewer.open() 

"""Update Logic"""
def bootstrap():
    try:
        if upd.check_and_update():
            import os, sys
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except:
        pass
    return


if __name__ == "__main__":
    if "--view" in sys.argv:
        sys.exit(create_viewer() or 0)

    if UPDATES:
        threading.Thread(target=bootstrap).start()
    app = QApplication(sys.argv)
    Viewer.apply_styles(app)
    icon_path = resolve_app_icon_path()
    if icon_path:
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    window = Window()
    if icon_path:
        window.setWindowIcon(app_icon)
    window.show()
    sys.exit(app.exec())