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
import _updates as upd

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
WHISPER_MODEL = "medium" #or "tiny", "base", "small", "large-v2"
WHISPER_BEAMSIZE = 5
WHISPER_PAT = 2
WHISPER_BESTOF = 3
GPU = False
UPDATES = True

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


def resolve_worker_command(worker_script):
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
    return os.path.dirname(os.path.dirname(__file__))


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

        self.gpu_input = QCheckBox("Enable GPU")
        self.gpu_input.setChecked(bool(settings["gpu"]))

        self.flatten_audio = QCheckBox("Flatten audio (pitch shift and bandpass, can help with alignment quality but is a lossy operation)")
        self.flatten_audio.setChecked(bool(settings.get("flatten", settings.get("flatten_audio", False))))

        self.phoneme_timestamps = QCheckBox("Phoneme-level timestamps (MFA alignment; richer karaoke highlighting)")
        self.phoneme_timestamps.setChecked(bool(settings.get("phoneme_timestamps", True)))
        try:
            subprocess.check_output(["nvidia-smi"], timeout=1)
            self.gpu_input.setEnabled(True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.gpu_input.setEnabled(False)
            self.gpu_input.setText("No compatible GPU detected (CPU only)")
        except subprocess.TimeoutExpired:
            self.gpu_input.setEnabled(False)
            self.gpu_input.setText("GPU check timed out (CPU only)")

        form.addRow(self.gpu_input)
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
        os.path.join(base_dir, "assets", "sonex0high-res"),
        os.path.join(base_dir, "assets", "sonex0high-res.png"),
        os.path.join(base_dir, "assets", "sonex-high-resolution-logo.png"),
        os.path.join(base_dir, "assets", "sonex-high-resolution-logo-transparent.png"),
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
        splash_path = os.path.join(base_dir, "assets", "sonex_splash.png")
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

        translation_group = QGroupBox("Transcription & Translation")
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
        self.translation_hint.setStyleSheet("color: #666666; font-size: 11px;")
        translation_layout.addRow(self.translation_hint)
        layout.addRow(translation_group)

        self.filebtn = QPushButton("Choose Media File (.MP3 Only)")
        self.filebtn.clicked.connect(self.on_want_file)
        layout.addRow(self.filebtn)

        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.open_advanced_settings)
        layout.addRow(self.advanced_button)


        
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

        worker_script = os.path.join(os.path.dirname(__file__), "_worker.py")
        program, base_args = resolve_worker_command(worker_script)
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
        dialog = AdvancedSettingsDialog(self.advanced_settings, self)
        if dialog.exec():
            self.advanced_settings = dialog.get_settings()

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

def bootstrap():
    try:
        if upd.check_and_update():
            import os, sys
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except:
        pass
    return

if __name__ == "__main__":
    if UPDATES:
        threading.Thread(target=bootstrap).start()
    app = QApplication(sys.argv)
    icon_path = resolve_app_icon_path()
    if icon_path:
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    window = Window()
    if icon_path:
        window.setWindowIcon(app_icon)
    window.show()
    sys.exit(app.exec())
