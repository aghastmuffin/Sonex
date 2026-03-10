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
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QProcess
from PyQt6.QtGui import QIcon
import sys, os, json, subprocess

language_dict = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german', 'it': 'italian', 'pt': 'portuguese', 'ru': 'russian', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean', 'ar': 'arabic', 'hi': 'hindi', 'bn': 'bengali', 'pa': 'punjabi', 'tr': 'turkish', 'vi': 'vietnamese', 'pl': 'polish', 'nl': 'dutch', 'sv': 'swedish', 'no': 'norwegian', 'da': 'danish', 'fi': 'finnish', 'he': 'hebrew', 'el': 'greek', 'th': 'thai', 'id': 'indonesian', 'uk': 'ukrainian', 'cs': 'czech', 'ro': 'romanian', 'hu': 'hungarian'}
#settings variables
# TODO: Hook them up
DEMUCS_MODEL = "htdemucs" #or "tasnet" or htdemucs 
DEMUCS_STEMS = "vocals" #or "other" or "both" or vocals
WHISPER_MODEL = "medium" #or "tiny", "base", "small", "large-v2"
WHISPER_BEAMSIZE = 5
WHISPER_PAT = 2
WHISPER_BESTOF = 3
GPU = False
WHISPER_TASK = "transcribe" #or "translate" !! Translate targets only English


class AdvancedSettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
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
        os.path.join(base_dir, "gui", "assets", "brand", "sonex0high-res"),
        os.path.join(base_dir, "gui", "assets", "brand", "sonex0high-res.png"),
        os.path.join(base_dir, "gui", "assets", "brand", "sonex-high-resolution-logo.png"),
        os.path.join(base_dir, "gui", "assets", "brand", "sonex-high-resolution-logo-transparent.png"),
    ]
    for icon_path in candidate_paths:
        if os.path.exists(icon_path):
            return icon_path
    return None


class PipelineWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, window, file_path, lang_code):
        super().__init__()
        self.window = window
        self.file_path = file_path
        self.lang_code = lang_code
        self.audiobase = None
        self.detected_lang = None
        

    @pyqtSlot()
    def run(self):
        try:
            self.progress.emit(5, "Starting...")
            self.audiobase, self.detected_lang = self.window.splitter(
                self.file_path,
                self.lang_code,
                progress_cb=self.progress.emit,
            )
            self.progress.emit(70, "Audio prep complete")
            self.window.notesanalysis(self.audiobase, progress_cb=self.progress.emit)
            self.progress.emit(100, "Done")
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

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

        self.setWindowTitle("Streamline GUI")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.lang_input = QComboBox(self)
        self.lang_input.setPlaceholderText("English")
        for code, name in language_dict.items():
            self.lang_input.addItem(f"{name.title()}", code)
        layout.addWidget(self.lang_input)

        

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

        self.worker_thread = None
        self.worker = None
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

        worker_script = os.path.join(os.path.dirname(__file__), "streamline_worker_process.py")
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
        else:
            self.set_progress(0, f"Error ({exit_code})")
        self.set_demucs_active(False)
        self.set_whisper_active(False)
        self.button.setEnabled(True)
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

    def splitter(self, file_path, lang_code=None, progress_cb=None):
        def normalize_lang_code(code):
            if not code:
                return None
            code = code.lower()
            if code in language_dict:
                return code
            for k, v in language_dict.items():
                if v == code:
                    return k
            return None
        

        progress = progress_cb if progress_cb else (lambda *_: None)
        from backbone.ltra.letra_toolkit import transcribe, align, separate
        progress(10, "Separating stems...")
        separate(file_path)
        audiobase = os.path.basename(file_path).removesuffix(".mp3")
        os.makedirs(audiobase, exist_ok=True)
        progress(25, "Transcribing vocals...")
        _, detectlang = transcribe(inp=f"{audiobase}/vocals.mp3", beam_size=WHISPER_BEAMSIZE, pat=WHISPER_PAT, best_of=WHISPER_BESTOF, outp=f"{audiobase}/vocals_whisper_segments.json", language=lang_code, model_size=WHISPER_MODEL, task=WHISPER_TASK) #TODO: allow advanced settings to set whisper custom settings, i.e model strength etc.
        #whisperx align
        progress(40, "Aligning words...")
        align(f"{audiobase}/vocals.mp3", f"{audiobase}/vocals_whisper_segments.json", f"{audiobase}/lyrics.txt", language=lang_code)
        if detectlang and not lang_code:
            lang_code = detectlang
        #mfa align
        try:
            progress(52, "Running MFA alignment...")
            from backbone.ltra import _mfa_aligner
            mfa_lang = detectlang or lang_code
            if mfa_lang in language_dict:
                _mfa_aligner.generate_aligned_v2(audiobase, acoustic=f"{language_dict[mfa_lang]}", dictionary=f"{language_dict[mfa_lang]}_mfa", allow_fuzzy=True, fuzzy_max_lookahead=8)
        except Exception as e:
            print("MFA Error:", e)
            pass
        from backbone.ltra.argos_translate import translate_file

        source_lang = normalize_lang_code(detectlang or lang_code or "es")
        target_lang = normalize_lang_code("en")

        if source_lang == target_lang:
            print(f"Skipping Argos translation: source and target are both '{target_lang}'.")
        else:
            progress(62, "Translating to English...")
            out = translate_file(
                f"{audiobase}/vocals_whisper_segments.json",
                from_lang=source_lang,
                to_lang=target_lang,
                verbose=True,
            )
        progress(68, "Text pipeline complete")
        return audiobase, (detectlang or lang_code)

    def notesanalysis(self, af, sr=48000, BEAT_STRENGTH_QUANTILE = 0.60, MIN_RELATIVE_BEAT_STRENGTH = 1.05, MIN_BEAT_GAP_MS = 120, BEAT_TOLERANCE_MS = 20, progress_cb=None):
        progress = progress_cb if progress_cb else (lambda *_: None)
        from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, SpectralPeaks, HPCP, RhythmExtractor2013
        import numpy as np
        def _score_beats(beats, bpm, confidence, duration_sec):
            if len(beats) < 2 or bpm <= 0:
                return -1e9
            intervals = np.diff(beats)
            mean_interval = float(np.mean(intervals))
            if mean_interval <= 0:
                return -1e9
            cv = float(np.std(intervals) / (mean_interval + 1e-9))
            stability = 1.0 / (1.0 + cv)
            expected_beats = duration_sec * bpm / 60.0
            coverage = 1.0 - abs(len(beats) - expected_beats) / (expected_beats + 1e-9)
            coverage = float(np.clip(coverage, 0.0, 1.0))
            conf = float(np.clip(confidence, 0.0, 1.0))
            return 0.5 * stability + 0.3 * coverage + 0.2 * conf
        def extract_best_rhythm(audio, duration_sec, sr):
            # Use HPSS to isolate percussive component for better beat detection
            import librosa
            print("Isolating percussive component for beat detection...")
            audio_np = audio.astype(np.float32)
            _, y_percussive = librosa.effects.hpss(audio_np)
            
            best = None
            for method in ("multifeature", "degara"):
                extractor = RhythmExtractor2013(method=method)
                # Run beat detection on percussive component only
                bpm, beats, confidence, estimates, bpm_intervals = extractor(y_percussive)
                score = _score_beats(beats, bpm, confidence, duration_sec)
                result = {
                    "method": method,
                    "bpm": bpm,
                    "beats": beats,
                    "confidence": confidence,
                    "estimates": estimates,
                    "bpm_intervals": bpm_intervals,
                    "score": score,
                }
                if best is None or result["score"] > best["score"]:
                    best = result
            return best

        def filter_beats_by_strength(audio, sr, beats,
                             strength_quantile=BEAT_STRENGTH_QUANTILE,
                             min_relative_strength=MIN_RELATIVE_BEAT_STRENGTH,
                             min_gap_ms=MIN_BEAT_GAP_MS):
            if len(beats) == 0:
                return beats, np.array([], dtype=np.float32), 0.0

            # Smoothed amplitude envelope as a simple beat-energy proxy
            env_window = max(1, int(round(0.050 * sr)))  # 50ms smoothing
            envelope = np.convolve(np.abs(audio), np.ones(env_window, dtype=np.float32) / env_window, mode='same')

            half_window = max(1, int(round(0.040 * sr)))  # +/-40ms around beat
            beat_strengths = np.zeros(len(beats), dtype=np.float32)
            for i, beat_time in enumerate(beats):
                center = int(round(float(beat_time) * sr))
                start = max(0, center - half_window)
                end = min(len(envelope), center + half_window + 1)
                if end > start:
                    beat_strengths[i] = float(np.max(envelope[start:end]))

            quantile_thr = float(np.quantile(beat_strengths, np.clip(strength_quantile, 0.0, 1.0)))
            song_ref = float(np.quantile(envelope, 0.75))
            relative_thr = song_ref * float(max(0.0, min_relative_strength))
            strength_threshold = max(quantile_thr, relative_thr)

            min_gap_sec = max(0.0, float(min_gap_ms) / 1000.0)
            filtered = []
            last_kept = -1e9
            for beat_time, strength in zip(beats, beat_strengths):
                bt = float(beat_time)
                if strength < strength_threshold:
                    continue
                if bt - last_kept < min_gap_sec:
                    continue
                filtered.append(bt)
                last_kept = bt

            return np.array(filtered, dtype=np.float32), beat_strengths, strength_threshold

        def save_novocs(af, fp, sr):
            file_path = fp
            AUDIO = af
            audio = MonoLoader(filename=file_path, sampleRate=sr)().astype(np.float32)

            # Normalize
            audio_max = np.max(np.abs(audio))
            if audio_max > 0:
                audio = audio / audio_max
            audio *= 20.0

            # Frame parameters
            frame_size = 4096
            hop_size = 48  # Exactly 1ms

            # Initialize Algorithms
            window = Windowing(type='hann')
            spectrum = Spectrum()
            spectral_peaks = SpectralPeaks(minFrequency=40, maxFrequency=5000, sampleRate=sr)
            hpcp_algo = HPCP(size=12, referenceFrequency=440.0,
                            minFrequency=40, maxFrequency=5000,
                            harmonics=8, bandPreset=False)

            print("Processing HPCP (Notes)...")
            frame_hpcps = []
            for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                if np.all(frame == 0):
                    frame_hpcps.append(np.zeros(12))
                    continue
                
                win_frame = window(frame)
                mag_spectrum = spectrum(win_frame)
                freqs, mags = spectral_peaks(mag_spectrum)
                
                if len(freqs) == 0:
                    frame_hpcps.append(np.zeros(12))
                    continue

                hpcp_frame = hpcp_algo(freqs, mags)
                
                # Normalize
                m_val = np.max(hpcp_frame)
                if m_val > 0:
                    hpcp_frame /= m_val
                frame_hpcps.append(hpcp_frame)

            frame_hpcps = np.array(frame_hpcps)
            num_frames = len(frame_hpcps)
            duration_sec = num_frames * hop_size / sr

            # --- Rhythm Extraction ---
            print("Extracting rhythm (BPM and Beats)...")
            # This returns timestamps in seconds
            rhythm = extract_best_rhythm(audio, duration_sec, sr)
            bpm = rhythm["bpm"]
            beats = rhythm["beats"]
            filtered_beats, beat_strengths, beat_strength_threshold = filter_beats_by_strength(audio, sr, beats)
            confidence = rhythm["confidence"]
            selected_method = rhythm["method"]

            # Create a beat map aligned to HPCP frame indices with tolerance for perceptual timing
            beat_map = np.zeros(num_frames, dtype=bool)
            beat_centers = np.zeros(num_frames, dtype=bool)
            beat_tolerance_ms = BEAT_TOLERANCE_MS
            beat_tolerance_frames = max(1, int(round((beat_tolerance_ms / 1000.0) * sr / hop_size)))
            for beat_time in filtered_beats:
                frame_index = int(round(beat_time * sr / hop_size))
                if 0 <= frame_index < num_frames:
                    beat_centers[frame_index] = True
                    start = max(0, frame_index - beat_tolerance_frames)
                    end = min(num_frames, frame_index + beat_tolerance_frames + 1)
                    beat_map[start:end] = True

            # --- Save Everything ---
            np.savez_compressed(f"{AUDIO}/{AUDIO}_novocs_analysis.npz", 
                                hpcp=frame_hpcps, 
                                beats=beat_map, 
                                beat_centers=beat_centers,
                                bpm=np.array([bpm]),
                                beat_times=filtered_beats,
                                beat_times_raw=beats,
                                beat_confidence=np.array([confidence]),
                                rhythm_method=np.array([selected_method]),
                                beat_strengths_raw=beat_strengths,
                                beat_strength_threshold=np.array([beat_strength_threshold]),
                                beat_strength_quantile=np.array([BEAT_STRENGTH_QUANTILE]),
                                min_relative_beat_strength=np.array([MIN_RELATIVE_BEAT_STRENGTH]),
                                min_beat_gap_ms=np.array([MIN_BEAT_GAP_MS]),
                                beat_tolerance_ms=np.array([beat_tolerance_ms]),
                                sample_rate=np.array([sr]),
                                frame_size=np.array([frame_size]),
                                hop_size=np.array([hop_size]))
            
        def savevocs(af, fp, sr):
            AUDIO = af
            file_path = fp

            audio = MonoLoader(filename=file_path, sampleRate=sr)().astype(np.float32)

            # Normalize
            audio_max = np.max(np.abs(audio))
            if audio_max > 0:
                audio = audio / audio_max
            audio *= 20.0

            # Frame parameters
            frame_size = 4096
            hop_size = 48  # Exactly 1ms

            # Initialize Algorithms
            window = Windowing(type='hann')
            spectrum = Spectrum()
            spectral_peaks = SpectralPeaks(minFrequency=40, maxFrequency=5000, sampleRate=sr)
            hpcp_algo = HPCP(size=12, referenceFrequency=440.0,
                            minFrequency=40, maxFrequency=5000,
                            harmonics=8, bandPreset=False)

            print("Processing HPCP (Notes)...")
            frame_hpcps = []
            for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                if np.all(frame == 0):
                    frame_hpcps.append(np.zeros(12))
                    continue
                
                win_frame = window(frame)
                mag_spectrum = spectrum(win_frame)
                freqs, mags = spectral_peaks(mag_spectrum)
                
                if len(freqs) == 0:
                    frame_hpcps.append(np.zeros(12))
                    continue

                hpcp_frame = hpcp_algo(freqs, mags)
                
                # Normalize
                m_val = np.max(hpcp_frame)
                if m_val > 0:
                    hpcp_frame /= m_val
                frame_hpcps.append(hpcp_frame)

            frame_hpcps = np.array(frame_hpcps)
            num_frames = len(frame_hpcps)
            duration_sec = num_frames * hop_size / sr

            # --- Rhythm Extraction ---
            print("Extracting rhythm (BPM and Beats)...")
            # This returns timestamps in seconds
            rhythm = extract_best_rhythm(audio, duration_sec, sr)
            bpm = rhythm["bpm"]
            beats = rhythm["beats"]
            filtered_beats, beat_strengths, beat_strength_threshold = filter_beats_by_strength(audio, sr, beats)
            confidence = rhythm["confidence"]
            selected_method = rhythm["method"]

            # Create a beat map aligned to HPCP frame indices with tolerance for perceptual timing
            beat_map = np.zeros(num_frames, dtype=bool)
            beat_centers = np.zeros(num_frames, dtype=bool)
            beat_tolerance_ms = BEAT_TOLERANCE_MS
            beat_tolerance_frames = max(1, int(round((beat_tolerance_ms / 1000.0) * sr / hop_size)))
            for beat_time in filtered_beats:
                frame_index = int(round(beat_time * sr / hop_size))
                if 0 <= frame_index < num_frames:
                    beat_centers[frame_index] = True
                    start = max(0, frame_index - beat_tolerance_frames)
                    end = min(num_frames, frame_index + beat_tolerance_frames + 1)
                    beat_map[start:end] = True

            # --- Save Everything ---
            np.savez_compressed(f"{AUDIO}/{AUDIO}_vocs_analysis.npz", 
                                hpcp=frame_hpcps, 
                                beats=beat_map, 
                                beat_centers=beat_centers,
                                bpm=np.array([bpm]),
                                beat_times=filtered_beats,
                                beat_times_raw=beats,
                                beat_confidence=np.array([confidence]),
                                rhythm_method=np.array([selected_method]),
                                beat_strengths_raw=beat_strengths,
                                beat_strength_threshold=np.array([beat_strength_threshold]),
                                beat_strength_quantile=np.array([BEAT_STRENGTH_QUANTILE]),
                                min_relative_beat_strength=np.array([MIN_RELATIVE_BEAT_STRENGTH]),
                                min_beat_gap_ms=np.array([MIN_BEAT_GAP_MS]),
                                beat_tolerance_ms=np.array([beat_tolerance_ms]),
                                sample_rate=np.array([sr]),
                                frame_size=np.array([frame_size]),
                                hop_size=np.array([hop_size]))
            

        progress(78, "Analyzing instrumental notes/beats...")
        file_path = f"{af}/htdemucs/{af}/no_vocals.mp3"
        file_path1 = f"{af}/vocals.mp3"
        save_novocs(af, file_path, sr)
        progress(87, "Analyzing vocals notes/beats...")
        savevocs(af, file_path1, sr)
        progress(96, "Analysis files saved")


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