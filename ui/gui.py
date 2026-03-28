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
)
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QIcon, QPixmap
import sys, os, json, subprocess
#TODO: Swap out the loading bar in the gui for a splash screen with a progress bar and status text.
language_dict = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german', 'it': 'italian', 'pt': 'portuguese', 'ru': 'russian', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean', 'ar': 'arabic', 'hi': 'hindi', 'bn': 'bengali', 'pa': 'punjabi', 'tr': 'turkish', 'vi': 'vietnamese', 'pl': 'polish', 'nl': 'dutch', 'sv': 'swedish', 'no': 'norwegian', 'da': 'danish', 'fi': 'finnish', 'he': 'hebrew', 'el': 'greek', 'th': 'thai', 'id': 'indonesian', 'uk': 'ukrainian', 'cs': 'czech', 'ro': 'romanian', 'hu': 'hungarian', None: "Find For Me"}
#settings variables
DEMUCS_MODEL = "htdemucs" #or "tasnet" or htdemucs 
DEMUCS_STEMS = "default" #or "other" or "both" or vocals
WHISPER_MODEL = "medium" #or "tiny", "base", "small", "large-v2"
WHISPER_BEAMSIZE = 5
WHISPER_PAT = 2
WHISPER_BESTOF = 3
GPU = False


def notify(title, message):
    CMD = '''
    on run argv
    display notification (item 2 of argv) with title (item 1 of argv)
    end run
    '''
    import platform
    if platform.system() == "Darwin":
        subprocess.call(['osascript', '-e', CMD, title, message])
    else:
        print("Notifications are currently not supported for this architecture. (Darwin-only)")

def dialogue(title, message):
    CMD = '''
    on run argv
    display dialogue (item 2 of argv) with title (item 1 of argv)
    end run
    '''
    subprocess.call(['osascript', '-e', CMD, title, message])


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
        self.whisper_model_input.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        self.whisper_model_input.setCurrentText(settings["whisper_model"])
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
        try: #Wisegpu
            subprocess.check_output(['nvidia-smi'], timeout=1)
            self.gpu_input.setEnabled(False)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.gpu_input.setEnabled(False)
            self.gpu_input.setText("No Compatiable GPU Detected, Using CPU")
        except subprocess.TimeoutExpired:
            self.gpu_input.setEnabled(False)
            self.gpu_input.setText("GPU Timed Out, Using CPU")

        form.addRow(self.gpu_input)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def get_settings(self):
        return {
            "demucs_model": self.demucs_model_input.currentText(),
            "demucs_stems": self.demucs_stems_input.currentText(),
            "whisper_model": self.whisper_model_input.currentText(),
            "whisper_beam_size": int(self.whisper_beam_input.value()),
            "whisper_patience": int(self.whisper_pat_input.value()),
            "whisper_best_of": int(self.whisper_bestof_input.value()),
            "gpu": bool(self.gpu_input.isChecked()),
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
        self.setWindowTitle("SONEX - Processing")
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
        }

        self.setWindowTitle("SONEX - Analyzer")
        self.setGeometry(100, 100, 400, 200)

        layout = QFormLayout()

        self.lang_input = QComboBox(self)
        self.lang_input.setPlaceholderText("From_Language")
        for code, name in language_dict.items():
            self.lang_input.addItem(f"{name.title()}", code)
        layout.addRow("From:", self.lang_input)

        self.lang_to = QComboBox(self)
        import locale
        syslang = locale.getlocale()[0][:2]
        self.lang_to.setPlaceholderText(str(language_dict[syslang]))
        for code, name in language_dict.items():
            self.lang_to.addItem(f"{name.title()}", code)
        layout.addRow("To:", self.lang_to)

        

        self.filebtn = QPushButton("Choose Media File (.MP3 Only)")
        self.filebtn.clicked.connect(self.on_want_file)
        layout.addRow(self.filebtn)

        self.advanced_button = QPushButton("Advanced Settings")
        self.advanced_button.clicked.connect(self.open_advanced_settings)

        self.translation_mode_input = QComboBox(self)
        self.translation_mode_input.addItem("None (keep source)", "none")
        self.translation_mode_input.addItem("Argos", "argos")
        self.translation_mode_input.addItem("Whisper", "whisper")
        self.translation_mode_input.addItem("Both", "both")
        layout.addRow(self.advanced_button, self.translation_mode_input)


        
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
        self.lang_code = self.lang_input.currentData()
        self.translation_mode = self.translation_mode_input.currentData() or "none"
        if not getattr(self, "file_path", None):
            return

        if self.pipeline_process is not None and self.pipeline_process.state() != QProcess.ProcessState.NotRunning:
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
        project_root = os.path.dirname(os.path.dirname(__file__))
        lang_arg = self.lang_code if self.lang_code else ""
        translation_mode_arg = self.translation_mode if self.translation_mode else "none"
        settings_arg = json.dumps(self.advanced_settings)

        self.pipeline_process = QProcess(self)
        self.pipeline_process.setWorkingDirectory(project_root)
        self.pipeline_process.setProgram(sys.executable)
        self.pipeline_process.setArguments([worker_script, self.file_path, lang_arg, translation_mode_arg, settings_arg, self.lang_to.currentData() or None])
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
        notify("Processing Complete", f"Your file has been processed. Output directory: {self.audiobase}" if self.audiobase else "Your file has been processed.")
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

if __name__ == "__main__":
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
