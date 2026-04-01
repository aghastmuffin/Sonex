import os
import sys
import traceback
import faulthandler
import json
from pathlib import Path

from sympy import flatten

faulthandler.enable()

language_dict = {
    'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german', 'it': 'italian', 'pt': 'portuguese',
    'ru': 'russian', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean', 'ar': 'arabic', 'hi': 'hindi',
    'bn': 'bengali', 'pa': 'punjabi', 'tr': 'turkish', 'vi': 'vietnamese', 'pl': 'polish', 'nl': 'dutch',
    'sv': 'swedish', 'no': 'norwegian', 'da': 'danish', 'fi': 'finnish', 'he': 'hebrew', 'el': 'greek',
    'th': 'thai', 'id': 'indonesian', 'uk': 'ukrainian', 'cs': 'czech', 'ro': 'romanian', 'hu': 'hungarian'
}


def default_output_root() -> Path:
    app_name = "Sonex"
    if sys.platform.startswith("darwin"):
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    return base / app_name / "outputs"



def emit_progress(value, label):
    print(f"PROGRESS|{int(value)}|{label}", flush=True)


def emit_demucs_active(is_active):
    print(f"DEMUCS_ACTIVE|{1 if is_active else 0}", flush=True)


def emit_demucs_progress(value, label="Demucs separating stems..."):
    print(f"DEMUCS_PROGRESS|{int(value)}|{label}", flush=True)


def emit_whisper_active(is_active):
    print(f"WHISPER_ACTIVE|{1 if is_active else 0}", flush=True)


def emit_whisper_progress(value, label="Whisper transcribing..."):
    print(f"WHISPER_PROGRESS|{int(value)}|{label}", flush=True)


def _stage_progress_cb(stage_start, stage_end, default_label, whisper_label=None):
    stage_start = int(stage_start)
    stage_end = int(stage_end)
    span = max(0, stage_end - stage_start)

    def _cb(value, label=None):
        try:
            clamped = max(0, min(100, int(value)))
        except (TypeError, ValueError):
            return
        mapped = stage_start + int((clamped / 100.0) * span)
        emit_progress(mapped, label or f"{default_label} {clamped}%")
        emit_whisper_progress(clamped, label or f"{whisper_label or default_label} {clamped}%")

    return _cb


def splitter(file_path, lang_code=None, translation_mode="none", settings=None, target_lang=None):
    # Ensure repository root is on sys.path before importing backbone modules.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root) #return to root dir so that it can begin to place files in proper place

    from backbone.ltra import letra_toolkit as lt
    from backbone.ltra.letra_toolkit import transcribe, align, separate
    from backbone.ltra.argos_translate import translate_file
    global demucs_stems

    settings = settings or {}
    demucs_model = settings.get("demucs_model", "htdemucs")
    demucs_stems = settings.get("demucs_stems", "default")
    whisper_model = settings.get("whisper_model", "medium")
    whisper_beam_size = int(settings.get("whisper_beam_size", 5))
    whisper_patience = int(settings.get("whisper_patience", 2))
    whisper_best_of = int(settings.get("whisper_best_of", 3))
    whisper_task = str(settings.get("whisper_task", "transcribe")).strip().lower()
    use_gpu = bool(settings.get("gpu", False))
    flattenaudio = bool(settings.get("flatten", False))
    wav2vec2_phone_fallback = bool(settings.get("wav2vec2_phone_fallback", False))
    wav2vec2_min_mfa_coverage = int(settings.get("wav2vec2_min_mfa_coverage", 85))

    if not use_gpu:
        # Force CPU path for demucs/whisper when GPU is disabled in advanced settings.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    lt.model = demucs_model
    lt.two_stems = None if demucs_stems in ("both", "default") else demucs_stems

    translation_mode = (translation_mode or "none").strip().lower()
    if translation_mode not in {"none", "argos", "whisper", "both"}:
        translation_mode = "none"
    emit_progress(10, "Separating stems...")
    emit_demucs_active(True)
    emit_demucs_progress(0, "Demucs separating stems...")
    try:
        separate(file_path, demucs_progress_cb=lambda value: emit_demucs_progress(value))
        emit_demucs_progress(100, "Demucs complete")
    finally:
        emit_demucs_active(False)

    audiobase = Path(file_path).stem
    os.makedirs(audiobase, exist_ok=True)
    

    if whisper_task == "translate":
        # Guard against legacy UI settings that would overwrite source transcripts with translated text.
        print("INFO|Ignoring legacy whisper_task=translate for primary transcript; using transcribe.", flush=True)

    emit_progress(25, "Transcribing vocals...")
    emit_whisper_active(True)
    emit_whisper_progress(0, "Whisper transcribing...")
    _, detectlang = transcribe(
        f"{audiobase}/vocals.mp3",
        whisper_beam_size,
        whisper_patience,
        whisper_best_of,
        f"{audiobase}/vocals_whisper_segments.json",
        language=lang_code,
        model_size=whisper_model,
        task="transcribe",
        progress_cb=_stage_progress_cb(25, 39, "Transcribing vocals...", whisper_label="Whisper transcribing..."),
    )
    emit_whisper_progress(100, "Whisper transcribing complete")
    emit_whisper_active(False)

    emit_progress(40, "Aligning words...")
    align(
        f"{audiobase}/vocals.mp3",
        f"{audiobase}/vocals_whisper_segments.json",
        f"{audiobase}/lyrics.txt",
        language=lang_code,
        flatten_audio=flattenaudio,
    )

    if detectlang and not lang_code:
        lang_code = detectlang

    try:
        emit_progress(52, "Running MFA alignment...")
        from backbone.ltra import _mfa_aligner
        mfa_lang = detectlang or lang_code
        if mfa_lang in language_dict:
            _mfa_aligner.generate_aligned_v2(
                audiobase,
                acoustic=f"{language_dict[mfa_lang]}",
                dictionary=f"{language_dict[mfa_lang]}",
                allow_fuzzy=True,
                fuzzy_max_lookahead=8,
            )

            if wav2vec2_phone_fallback:
                try:
                    current_cov = _mfa_aligner.phone_word_coverage(f"{audiobase}/mfa_vocals_phone_segments.json")
                    print(
                        f"wav2vec2 fallback check: MFA phone coverage={current_cov:.2f}% (min={wav2vec2_min_mfa_coverage}%)",
                        flush=True,
                    )
                    if current_cov < float(wav2vec2_min_mfa_coverage):
                        emit_progress(55, "Running wav2vec2 phone fallback...")
                        align(
                            f"{audiobase}/vocals.mp3",
                            f"{audiobase}/vocals_whisper_segments.json",
                            f"{audiobase}/vocals_whisper_segments_wav2vec2.json",
                            language=lang_code,
                            reuse_existing=False,
                            return_char_alignments=True,
                        )
                        stats = _mfa_aligner.fill_missing_phones_from_char_alignments(
                            audiobase,
                            aligned_chars_json="vocals_whisper_segments_wav2vec2.json",
                            phone_json="mfa_vocals_phone_segments.json",
                            base_json="mfa_vocals_whisper_segments.json",
                            out_json="mfa_vocals_phone_segments.json",
                        )
                        print(
                            f"wav2vec2 fallback filled {stats['filled_words']} words; coverage now {stats['coverage_after']:.2f}%",
                            flush=True,
                        )
                except Exception as fallback_exc:
                    print(f"wav2vec2 fallback error: {fallback_exc}", flush=True)
    except Exception as e:
        print(f"MFA Error: {e}", flush=True)

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
    try:
        import locale
        syslang = locale.getlocale()[0][:2]
        normalize_lang_code(syslang)  # Test if we can normalize the system language code
    except:
        syslang = "en"

    source_lang = normalize_lang_code(detectlang or lang_code or syslang)
    target_lang = normalize_lang_code(target_lang or syslang) #XXX

    if translation_mode in {"whisper", "both"}:
        emit_progress(58, "Whisper translation pass...")
        emit_whisper_active(True)
        emit_whisper_progress(0, "Whisper translating...")
        whisper_out = f"{audiobase}/whisper_translated.json"
        transcribe(
            f"{audiobase}/vocals.mp3",
            whisper_beam_size,
            whisper_patience,
            whisper_best_of,
            whisper_out,
            language=(detectlang or lang_code),
            model_size=whisper_model,
            task="translate",
            reuse_existing=False,
            progress_cb=_stage_progress_cb(58, 61, "Whisper translation pass...", whisper_label="Whisper translating..."),
        )
        print(f"INFO|Whisper translation saved to {whisper_out}", flush=True)
        emit_whisper_progress(100, "Whisper translation complete")
        emit_whisper_active(False)

    if translation_mode in {"argos", "both"}:
        if source_lang == target_lang:
            print(f"Skipping Argos translation: source and target are both '{target_lang}'.", flush=True)
        else:
            emit_progress(62, "Argos translation pass...")
            argos_out = f"{audiobase}/argos_translated.json"
            translate_file(
                f"{audiobase}/vocals_whisper_segments.json",
                # Keep source-language transcript unchanged and write Argos output separately.
                output_path=argos_out,
                from_lang=source_lang,
                to_lang=target_lang,
                verbose=True,
            )
            print(f"INFO|Argos translation saved to {argos_out}", flush=True)

    emit_progress(68, "Text pipeline complete")
    return str((Path.cwd() / audiobase).resolve()), (detectlang or lang_code)


def notesanalysis(af, output_root: Path, sr=48000, beat_strength_quantile=0.60, min_relative_beat_strength=1.05,
                  min_beat_gap_ms=120, beat_tolerance_ms=20):
    from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, SpectralPeaks, HPCP, RhythmExtractor2013
    import numpy as np

    NOTE_NAMES = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    af_path = Path(af)
    audio_base = af_path.name

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

    def extract_best_rhythm(audio, duration_sec, sr_local):
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

    def detect_beats_from_energy_spikes(audio, sr_local, min_gap_ms_local):
        """Drum-only beat detection using drummervisual-style normalized RMS thresholding."""
        import librosa
        from scipy.signal import find_peaks

        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(
            y=np.asarray(audio, dtype=np.float32),
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        if len(rms) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0.60, 0.0

        rms_max = float(np.max(rms))
        if rms_max <= 1e-9:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0.60, 0.0

        rms_norm = rms / (rms_max + 1e-9)
        # Same spirit as drummervisual: fixed normalized RMS threshold.
        strength_threshold = 0.60
        min_gap_frames = max(1, int(round((min_gap_ms_local / 1000.0) * sr_local / hop_length)))
        peak_indices, _ = find_peaks(rms_norm, height=strength_threshold, distance=min_gap_frames)

        beat_times = librosa.frames_to_time(
            peak_indices,
            sr=sr_local,
            hop_length=hop_length,
        ).astype(np.float32)
        beat_strengths = rms_norm[peak_indices].astype(np.float32)

        # Estimate BPM from median inter-onset interval
        if len(beat_times) >= 2:
            median_interval = float(np.median(np.diff(beat_times)))
            bpm = 60.0 / median_interval if median_interval > 0 else 0.0
        else:
            bpm = 0.0

        return beat_times, beat_strengths, strength_threshold, float(bpm)

    def filter_beats_by_strength(audio, sr_local, beats):
        if len(beats) == 0:
            return beats, np.array([], dtype=np.float32), 0.0

        env_window = max(1, int(round(0.050 * sr_local)))
        envelope = np.convolve(np.abs(audio), np.ones(env_window, dtype=np.float32) / env_window, mode='same')

        half_window = max(1, int(round(0.040 * sr_local)))
        beat_strengths = np.zeros(len(beats), dtype=np.float32)
        for i, beat_time in enumerate(beats):
            center = int(round(float(beat_time) * sr_local))
            start = max(0, center - half_window)
            end = min(len(envelope), center + half_window + 1)
            if end > start:
                beat_strengths[i] = float(np.max(envelope[start:end]))

        quantile_thr = float(np.quantile(beat_strengths, np.clip(beat_strength_quantile, 0.0, 1.0)))
        song_ref = float(np.quantile(envelope, 0.75))
        relative_thr = song_ref * float(max(0.0, min_relative_beat_strength))
        strength_threshold = max(quantile_thr, relative_thr)

        min_gap_sec = max(0.0, float(min_beat_gap_ms) / 1000.0)
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

    def analyze_and_save(audio_name, file_path, out_name, rhythm_file_path=None, is_drums_only=False):
        audio = MonoLoader(filename=file_path, sampleRate=sr)().astype(np.float32)
        rhythm_audio = MonoLoader(filename=(rhythm_file_path or file_path), sampleRate=sr)().astype(np.float32)

        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            audio = audio / audio_max
        audio *= 20.0

        rhythm_max = np.max(np.abs(rhythm_audio))
        if rhythm_max > 0:
            rhythm_audio = rhythm_audio / rhythm_max
        rhythm_audio *= 20.0

        frame_size = 4096
        hop_size = 48

        window = Windowing(type='hann')
        spectrum = Spectrum()
        spectral_peaks = SpectralPeaks(minFrequency=40, maxFrequency=5000, sampleRate=sr)
        hpcp_algo = HPCP(
            size=12,
            referenceFrequency=440.0,
            minFrequency=40,
            maxFrequency=5000,
            harmonics=8,
            bandPreset=False,
        )

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
            m_val = np.max(hpcp_frame)
            if m_val > 0:
                hpcp_frame /= m_val
            frame_hpcps.append(hpcp_frame)

        frame_hpcps = np.array(frame_hpcps)
        # Keep explicit note-strength views in addition to raw HPCP frames.
        # `frame_hpcps` already stores per-frame pitch-class strength (12 bins).
        note_strengths_per_frame = frame_hpcps.astype(np.float32)
        note_strengths_sum = np.sum(note_strengths_per_frame, axis=0).astype(np.float32)
        note_strengths_mean = np.mean(note_strengths_per_frame, axis=0).astype(np.float32)
        note_strengths_max = np.max(note_strengths_per_frame, axis=0).astype(np.float32)
        total_strength = float(np.sum(note_strengths_sum))
        if total_strength > 0.0:
            note_strengths_distribution = (note_strengths_sum / total_strength).astype(np.float32)
        else:
            note_strengths_distribution = np.zeros(12, dtype=np.float32)

        dominant_note_index_per_frame = np.argmax(note_strengths_per_frame, axis=1).astype(np.int16)
        dominant_note_strength_per_frame = np.max(note_strengths_per_frame, axis=1).astype(np.float32)

        num_frames = len(frame_hpcps)
        duration_sec = len(rhythm_audio) / sr

        if is_drums_only:
            # Drum stem: use drummervisual-like RMS threshold beat detection.
            print("Drum stem detected — using RMS-threshold beat detection...")
            filtered_beats, beat_strengths, beat_strength_threshold, bpm = \
                detect_beats_from_energy_spikes(rhythm_audio, sr, min_beat_gap_ms)
            beats = filtered_beats  # raw == filtered for npz consistency
            confidence = 1.0
            selected_method = "rms_visual"
        else:
            rhythm = extract_best_rhythm(rhythm_audio, duration_sec, sr)
            bpm = rhythm["bpm"]
            beats = rhythm["beats"]
            filtered_beats, beat_strengths, beat_strength_threshold = filter_beats_by_strength(rhythm_audio, sr, beats)
            confidence = rhythm["confidence"]
            selected_method = rhythm["method"]

        beat_map = np.zeros(num_frames, dtype=bool)
        beat_centers = np.zeros(num_frames, dtype=bool)
        beat_tolerance_frames = max(1, int(round((beat_tolerance_ms / 1000.0) * sr / hop_size)))
        for beat_time in filtered_beats:
            frame_index = int(round(beat_time * sr / hop_size))
            if 0 <= frame_index < num_frames:
                beat_centers[frame_index] = True
                start = max(0, frame_index - beat_tolerance_frames)
                end = min(num_frames, frame_index + beat_tolerance_frames + 1)
                beat_map[start:end] = True
        out_path = output_root / f"{audio_name}_{out_name}.npz"
        np.savez_compressed(
            out_path,
            hpcp=frame_hpcps,
            note_names=NOTE_NAMES,
            note_strengths_per_frame=note_strengths_per_frame,
            note_strengths_sum=note_strengths_sum,
            note_strengths_mean=note_strengths_mean,
            note_strengths_max=note_strengths_max,
            note_strengths_distribution=note_strengths_distribution,
            dominant_note_index_per_frame=dominant_note_index_per_frame,
            dominant_note_strength_per_frame=dominant_note_strength_per_frame,
            beats=beat_map,
            beat_centers=beat_centers,
            bpm=np.array([bpm]),
            beat_times=filtered_beats,
            beat_times_raw=beats,
            beat_confidence=np.array([confidence]),
            rhythm_method=np.array([selected_method]),
            beat_strengths_raw=beat_strengths,
            beat_strength_threshold=np.array([beat_strength_threshold]),
            beat_strength_quantile=np.array([beat_strength_quantile]),
            min_relative_beat_strength=np.array([min_relative_beat_strength]),
            min_beat_gap_ms=np.array([min_beat_gap_ms]),
            beat_tolerance_ms=np.array([beat_tolerance_ms]),
            sample_rate=np.array([sr]),
            frame_size=np.array([frame_size]),
            hop_size=np.array([hop_size]),
        )

    emit_progress(78, "Analyzing instrumental notes/beats...")
    using_drum_stem = demucs_stems in ("both", "default")
    if using_drum_stem:
        novocal_path = f"{af}/htdemucs/{audio_base}/drums.mp3"
        melodic_path = f"{af}/other.mp3"
        if not os.path.exists(melodic_path):
            melodic_path = f"{af}/htdemucs/{audio_base}/other.mp3"
    else:
        novocal_path = f"{af}/htdemucs/{audio_base}/no_vocals.mp3"
        # Keep old behavior for non-full-stem runs (e.g., vocals): one source for both rhythm and melodic.
        melodic_path = novocal_path

    analyze_and_save(audio_base, melodic_path, "novocs_analysis", rhythm_file_path=novocal_path, is_drums_only=using_drum_stem)

    emit_progress(87, "Analyzing vocals notes/beats...")
    vocal_path = f"{af}/vocals.mp3"
    analyze_and_save(audio_base, vocal_path, "vocs_analysis")

    emit_progress(96, "Analysis files saved")


def main():
    if len(sys.argv) < 2:
        print("ERROR|Missing input file path", flush=True)
        return 2

    file_path = sys.argv[1]
    lang_code = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    lang_code_to = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
    output_root_arg = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else None
    translation_mode = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "none"
    raw_settings = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "{}"

    output_root = Path(output_root_arg) if output_root_arg else default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(output_root)

    try:
        settings = json.loads(raw_settings)
    except json.JSONDecodeError:
        settings = {}

    try:
        emit_progress(5, "Starting...")
        audiobase, detected_lang = splitter(
            file_path,
            lang_code=lang_code,
            target_lang=lang_code_to,
            translation_mode=translation_mode,
            settings=settings,
        )
        print(f"AUDIOBASE|{audiobase}", flush=True)
        if detected_lang:
            print(f"LANG|{detected_lang}", flush=True)
        emit_progress(70, "Audio prep complete")
        notesanalysis(audiobase, output_root=output_root)
        emit_progress(100, "Done")
        return 0
    except Exception as exc:
        print(f"ERROR|{exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("You've ran a worker process! This is not meant to be run directly. If you're seeing this, something went wrong.", flush=True)
    sys.exit(main())
