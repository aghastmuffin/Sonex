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
from PyQt6.QtCore import QProcess
from PyQt6.QtGui import QIcon
import sys, os, json, subprocess
#TODO: Swap out the loading bar in the gui for a splash screen with a progress bar and status text.
language_dict = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german', 'it': 'italian', 'pt': 'portuguese', 'ru': 'russian', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean', 'ar': 'arabic', 'hi': 'hindi', 'bn': 'bengali', 'pa': 'punjabi', 'tr': 'turkish', 'vi': 'vietnamese', 'pl': 'polish', 'nl': 'dutch', 'sv': 'swedish', 'no': 'norwegian', 'da': 'danish', 'fi': 'finnish', 'he': 'hebrew', 'el': 'greek', 'th': 'thai', 'id': 'indonesian', 'uk': 'ukrainian', 'cs': 'czech', 'ro': 'romanian', 'hu': 'hungarian', None: "Find For Me"}
#settings variables
DEMUCS_MODEL = "htdemucs" #or "tasnet" or htdemucs 
DEMUCS_STEMS = "vocals" #or "other" or "both" or vocals
WHISPER_MODEL = "medium" #or "tiny", "base", "small", "large-v2"
WHISPER_BEAMSIZE = 5
WHISPER_PAT = 2
WHISPER_BESTOF = 3
GPU = False
WHISPER_TASK = "transcribe" #or "translate" !! Translate targets only English


def notify(title, message):
    CMD = '''
    on run argv
    display notification (item 2 of argv) with title (item 1 of argv)
    end run
    '''
    subprocess.call(['osascript', '-e', CMD, title, message])

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

        self.whisper_task_input = QComboBox(self)
        self.whisper_task_input.addItems(["transcribe", "translate"])
        self.whisper_task_input.setCurrentText(settings["whisper_task"])
        form.addRow("Whisper task", self.whisper_task_input)

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
            "whisper_task": self.whisper_task_input.currentText(),
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
            "whisper_task": WHISPER_TASK,
            "gpu": GPU,
        }

        self.setWindowTitle("SONEX - Analyzer")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.lang_input = QComboBox(self)
        self.lang_input.setPlaceholderText("From_Language")
        for code, name in language_dict.items():
            self.lang_input.addItem(f"{name.title()}", code)
        layout.addWidget(self.lang_input)

        self.lang_to = QComboBox(self)
        self.lang_to.setPlaceholderText("To_Language (Translation)")
        for code, name in language_dict.items():
            self.lang_to.addItem(f"{name.title()}", code)
        layout.addWidget(self.lang_to)

        

        self.filebtn = QPushButton("Choose Media File (.MP3 Only)")
        self.filebtn.clicked.connect(self.on_want_file)
        layout.addWidget(self.filebtn)

        config_holder = QFormLayout()
        self.advanced_button = QPushButton("Advanced")
        self.advanced_button.clicked.connect(self.open_advanced_settings)

        self.translation_mode_input = QComboBox(self)
        self.translation_mode_input.addItem("Argos (default)", "argos")
        self.translation_mode_input.addItem("Whisper", "whisper")
        self.translation_mode_input.addItem("Both", "both")
        config_holder.addRow(self.advanced_button, self.translation_mode_input)

        layout.addLayout(config_holder)

        
        self.button = QPushButton("Choose File First")
        self.button.setEnabled(False)
        self.button.clicked.connect(self.on_button_click)
        layout.addWidget(self.button)
        
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

        layout.addWidget(self.prog_bar)
        layout.addWidget(self.demucs_prog_bar)
        layout.addWidget(self.whisper_prog_bar)

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
        self.translation_mode = self.translation_mode_input.currentData() or "argos"
        if not getattr(self, "file_path", None):
            return

        if self.pipeline_process is not None and self.pipeline_process.state() != QProcess.ProcessState.NotRunning:
            return

        self.button.setEnabled(False)
        self.set_progress(5, "Starting...")
        self.set_demucs_active(False)
        self.set_whisper_active(False)

        worker_script = os.path.join(os.path.dirname(__file__), "_worker.py")
        lang_arg = self.lang_code if self.lang_code else ""
        translation_mode_arg = self.translation_mode if self.translation_mode else "argos"
        settings_arg = json.dumps(self.advanced_settings)

        self.pipeline_process = QProcess(self)
        self.pipeline_process.setWorkingDirectory(os.path.dirname(__file__))
        self.pipeline_process.setProgram(sys.executable)
        self.pipeline_process.setArguments([worker_script, self.file_path, lang_arg, translation_mode_arg, settings_arg])
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
                continue
            if line.startswith("DEMUCS_PROGRESS|"):
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    try:
                        progress_value = int(parts[1])
                        label = parts[2] if len(parts) == 3 else None
                        self.set_demucs_progress(progress_value, label)
                    except ValueError:
                        pass
                continue
            if line.startswith("WHISPER_ACTIVE|"):
                parts = line.split("|", 1)
                if len(parts) == 2:
                    self.set_whisper_active(parts[1] == "1")
                continue
            if line.startswith("WHISPER_PROGRESS|"):
                parts = line.split("|", 2)
                if len(parts) >= 2:
                    try:
                        progress_value = int(parts[1])
                        label = parts[2] if len(parts) == 3 else None
                        self.set_whisper_progress(progress_value, label)
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
