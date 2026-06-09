"""Sonex worker process — runs the audio pipeline outside the GUI.

Spawned by Sonex.py via QProcess (or as a frozen sonex-worker binary). Communicates
back to the parent through tagged stdout lines that the GUI parses for progress bars
and status updates.

Stdout protocol (parent reads these prefixes):
    PROGRESS|<0-100>|<label>
    DEMUCS_ACTIVE|0|1
    DEMUCS_PROGRESS|<0-100>|<label>
    WHISPER_ACTIVE|0|1
    WHISPER_PROGRESS|<0-100>|<label>
    AUDIOBASE|<resolved output folder>
    LANG|<detected iso code>
    INFO|<message>
    ERROR|<message>

CLI arguments (see main()):
    argv[1]  input file path
    argv[2]  source language (or detect sentinel)
    argv[3]  translation mode: none | argos | whisper | both
    argv[4]  advanced settings JSON
    argv[5]  target language
    argv[6]  output root directory
"""
import os
import shutil
import sys
import traceback
import faulthandler
import json
from pathlib import Path

# Dump Python tracebacks on crash (useful when the worker is a separate process).
faulthandler.enable()


def _load_language_helpers():
    """Lazy-import language helpers so the worker starts fast before heavy imports."""
    from backbone.ltra.languages import (
        MFA_LANGUAGE_NAMES,
        get_system_language_code,
        normalize_lang_code,
        resolve_source_language,
        resolve_target_language,
    )

    return (
        MFA_LANGUAGE_NAMES,
        get_system_language_code,
        normalize_lang_code,
        resolve_source_language,
        resolve_target_language,
    )


def default_output_root() -> Path:
    """Platform-specific default for processed output (delegates to frase_core)."""
    from ui.frase_core import default_output_root as _default_root
    return Path(_default_root())



# ---------------------------------------------------------------------------
# Progress emitters — stdout lines consumed by Sonex.py::Window::on_pipeline_stdout
# ---------------------------------------------------------------------------

def emit_progress(value, label):
    """Emit overall pipeline progress (0–100) to the parent GUI."""
    print(f"PROGRESS|{int(value)}|{label}", flush=True)


def emit_demucs_active(is_active):
    """Toggle visibility of the Demucs sub-progress bar in the GUI."""
    print(f"DEMUCS_ACTIVE|{1 if is_active else 0}", flush=True)


def emit_demucs_progress(value, label="Demucs separating stems..."):
    """Emit Demucs stem-separation progress (0–100)."""
    print(f"DEMUCS_PROGRESS|{int(value)}|{label}", flush=True)


def emit_whisper_active(is_active):
    """Toggle visibility of the Whisper sub-progress bar in the GUI."""
    print(f"WHISPER_ACTIVE|{1 if is_active else 0}", flush=True)


def emit_whisper_progress(value, label="Whisper transcribing..."):
    """Emit Whisper transcription/translation progress (0–100)."""
    print(f"WHISPER_PROGRESS|{int(value)}|{label}", flush=True)


def _has_phone_level_data(json_path: Path, min_coverage: float = 0.85) -> bool:
    """Return True when enough words in a transcript JSON have valid phone timings."""
    try:
        if not json_path.exists():
            return False
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(data, list):
        return False

    total_words = 0
    words_with_phones = 0
    for seg in data:
        for w in seg.get("words", []) or []:
            total_words += 1
            valid = False
            for p in w.get("phones", []) or []:
                try:
                    ps = float(p.get("start"))
                    pe = float(p.get("end"))
                    if pe > ps:
                        valid = True
                        break
                except (TypeError, ValueError, KeyError):
                    continue
            if not valid:
                for ps in w.get("phone_segments", []) or []:
                    try:
                        pstart = float(ps.get("start"))
                        pend = float(ps.get("end"))
                        if pend > pstart:
                            valid = True
                            break
                    except (TypeError, ValueError, KeyError):
                        continue
            if valid:
                words_with_phones += 1

    if total_words <= 0:
        return False

    coverage = float(words_with_phones) / float(total_words)
    return coverage >= float(min_coverage)


def _has_dual_transcript_data(json_path: Path) -> bool:
    """Return True when a transcript JSON contains both source and target text."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(data, list):
        return False

    for seg in data:
        has_source = bool(seg.get("source_words")) or bool((seg.get("source_text") or "").strip())
        has_target = bool(seg.get("words")) or bool((seg.get("text") or "").strip())
        if has_source and has_target:
            return True
    return False


def _enrich_translated_source_phones(audiobase: str):
    """Copy MFA phoneme timings onto translated.json source_words for dual+phoneme playback."""
    base = Path(audiobase)
    phone_json = base / "mfa_vocals_phone_segments.json"
    trans_json = base / "translated.json"
    if not phone_json.exists() or not trans_json.exists():
        return

    try:
        phone_segs = json.loads(phone_json.read_text(encoding="utf-8"))
        trans_segs = json.loads(trans_json.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"INFO|Phoneme enrich skipped: {exc}", flush=True)
        return

    if not isinstance(phone_segs, list) or not isinstance(trans_segs, list):
        return

    enriched_words = 0
    for seg_idx, tseg in enumerate(trans_segs):
        if seg_idx >= len(phone_segs):
            break
        phone_words = phone_segs[seg_idx].get("words") or []
        source_words = tseg.get("source_words")
        if not isinstance(source_words, list) or not source_words:
            continue

        updated = []
        for word_idx, word in enumerate(source_words):
            merged = dict(word)
            if word_idx < len(phone_words):
                for key in ("phones", "phone_segments", "speaker"):
                    if phone_words[word_idx].get(key):
                        merged[key] = phone_words[word_idx][key]
                        enriched_words += 1
            updated.append(merged)
        tseg["source_words"] = updated

    try:
        trans_json.write_text(json.dumps(trans_segs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"INFO|Attached MFA phoneme timings to translated source_words ({enriched_words} fields)", flush=True)
    except Exception as exc:
        print(f"INFO|Phoneme enrich write skipped: {exc}", flush=True)


def _write_playback_transcript(audiobase: str, settings=None, translation_mode: str = "none"):
    """Pick the best transcript for the lyrics viewer and copy it to playback_segments.json."""
    base = Path(audiobase)
    phone_json = base / "mfa_vocals_phone_segments.json"
    word_json = base / "vocals_whisper_segments.json"
    playback_json = base / "playback_segments.json"

    settings = settings or {}
    phoneme_timestamps = bool(settings.get("phoneme_timestamps", True))

    source = None
    if phoneme_timestamps and _has_phone_level_data(phone_json):
        source = phone_json
    elif word_json.exists():
        source = word_json
    elif phone_json.exists():
        source = phone_json

    if source is None or not source.exists():
        return

    try:
        shutil.copy2(source, playback_json)
        print(f"INFO|Playback transcript: {source.name}", flush=True)
    except Exception as exc:
        # Silent fallback behavior: don't fail pipeline on playback copy issues.
        print(f"INFO|Playback transcript preflight skipped: {exc}", flush=True)


def _stage_progress_cb(stage_start, stage_end, default_label, whisper_label=None):
    """Map a sub-stage's 0–100 progress into the overall pipeline percentage range."""
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
    """Run the full lyrics pipeline: separate → transcribe → align → translate.

    Stages (approximate overall progress):
        10%  Demucs stem separation
        25%  Whisper transcription
        40%  WhisperX word alignment
        48%  Speaker diarization (optional)
        52%  MFA phoneme alignment (optional)
        58%  Whisper translation pass (optional)
        62%  OpusMT/NLLB translation (optional)
        68%  Playback transcript preflight

    Returns:
        (audiobase_path, detected_language_code)
    """
    # Ensure repository root is on sys.path before importing backbone modules.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root) #return to root dir so that it can begin to place files in proper place

    from backbone.ltra import letra_toolkit as lt
    from backbone.ltra.letra_toolkit import transcribe, align, diarize, separate
    from backbone.ltra.argos_translate import translate_file
    (
        mfa_language_names,
        get_system_language_code,
        normalize_lang_code,
        resolve_source_language,
        resolve_target_language,
    ) = _load_language_helpers()
    global demucs_stems  # read later by notesanalysis() for stem path resolution

    settings = settings or {}
    demucs_model = settings.get("demucs_model", "htdemucs")
    demucs_stems = settings.get("demucs_stems", "default")
    from backbone.ltra.whisper_models import normalize_whisper_model_name

    whisper_model = normalize_whisper_model_name(settings.get("whisper_model", "medium"))
    whisper_beam_size = int(settings.get("whisper_beam_size", 5))
    whisper_patience = int(settings.get("whisper_patience", 2))
    whisper_best_of = int(settings.get("whisper_best_of", 3))
    whisper_task = str(settings.get("whisper_task", "transcribe")).strip().lower()
    use_gpu = bool(settings.get("gpu", False))
    flattenaudio = bool(settings.get("flatten", False))
    phoneme_timestamps = bool(settings.get("phoneme_timestamps", True))
    speaker_diarization = bool(settings.get("speaker_diarization", False))
    wav2vec2_phone_fallback = bool(settings.get("wav2vec2_phone_fallback", False))
    wav2vec2_min_mfa_coverage = int(settings.get("wav2vec2_min_mfa_coverage", 85))

    if not use_gpu:
        # Force CPU path for demucs/whisper when GPU is disabled in advanced settings.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Configure letra_toolkit globals before calling separate/transcribe.
    lt.model = demucs_model
    lt.two_stems = None if demucs_stems in ("both", "default") else demucs_stems

    translation_mode = (translation_mode or "none").strip().lower()
    if translation_mode not in {"none", "argos", "whisper", "both"}:
        translation_mode = "none"

    lang_code = resolve_source_language(lang_code)
    target_lang = resolve_target_language(target_lang)

    # --- Stage 1: Demucs stem separation ---
    emit_progress(10, "Separating stems...")
    emit_demucs_active(True)
    emit_demucs_progress(0, "Demucs separating stems...")
    try:
        separate(file_path, demucs_progress_cb=lambda value: emit_demucs_progress(value))
        emit_demucs_progress(100, "Demucs complete")
    finally:
        emit_demucs_active(False)

    # Output folder named after the input file stem (e.g. "mysong/" for "mysong.mp3").
    audiobase = Path(file_path).stem
    os.makedirs(audiobase, exist_ok=True)

    if whisper_task == "translate":
        # Guard against legacy UI settings that would overwrite source transcripts with translated text.
        print("INFO|Ignoring legacy whisper_task=translate for primary transcript; using transcribe.", flush=True)

    # --- Stage 2: Whisper transcription (native language) ---
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

    # Prefer detected language from Whisper when caller did not provide one.
    if detectlang and not lang_code:
        lang_code = detectlang

    # --- Stage 3: WhisperX CTC word-level alignment ---
    emit_progress(40, "Aligning words...")
    align(
        f"{audiobase}/vocals.mp3",
        f"{audiobase}/vocals_whisper_segments.json",
        f"{audiobase}/lyrics.txt",
        language=(lang_code or detectlang or "en"),
        flatten_audio=flattenaudio,
    )

    # --- Stage 4: Speaker diarization (optional, requires HF_TOKEN) ---
    if speaker_diarization:
        emit_progress(48, "Diarizing speakers...")
        try:
            diarize(
                f"{audiobase}/vocals.mp3",
                f"{audiobase}/vocals_whisper_segments.json",
                progress_cb=_stage_progress_cb(48, 51, "Diarizing speakers..."),
            )
        except Exception as exc:
            print(f"INFO|Speaker diarization skipped: {exc}", flush=True)
    else:
        print("INFO|Speaker diarization disabled; skipping.", flush=True)

    # --- Stage 5: MFA phoneme alignment (optional) ---
    if phoneme_timestamps:
        try:
            emit_progress(52, "Running MFA alignment...")
            from backbone.ltra import _mfa_aligner
            mfa_lang = detectlang or lang_code
            if mfa_lang in mfa_language_names:
                _mfa_aligner.generate_aligned_v2(
                    audiobase,
                    acoustic=f"{mfa_language_names[mfa_lang]}",
                    dictionary=f"{mfa_language_names[mfa_lang]}",
                    allow_fuzzy=True,
                    fuzzy_max_lookahead=8,
                    phonemizer_language=mfa_lang,
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
                                phonemizer_language=mfa_lang,
                            )
                            print(
                                f"wav2vec2 fallback filled {stats['filled_words']} words; fallback words {stats['fallback_words']}; coverage now {stats['coverage_after']:.2f}%",
                                flush=True,
                            )
                    except Exception as fallback_exc:
                        print(f"wav2vec2 fallback error: {fallback_exc}", flush=True)
        except Exception as e:
            print(f"INFO|Phone-level alignment unavailable; defaulting to word-level: {e}", flush=True)
    else:
        print("INFO|Phoneme timestamps disabled; skipping MFA phone alignment.", flush=True)

    # --- Stage 6: Translation passes (optional) ---
    source_lang = normalize_lang_code(detectlang or lang_code) or get_system_language_code()
    target_lang = resolve_target_language(target_lang)

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
            print(f"Skipping translation: source and target are both '{target_lang}'.", flush=True)
        else:
            emit_progress(62, f"Translation pass (OpusMT/NLLB -> {target_lang})...")
            trans_out = f"{audiobase}/translated.json"
            translate_file(
                f"{audiobase}/vocals_whisper_segments.json",
                output_path=trans_out,
                from_lang=source_lang,
                to_lang=target_lang,
                verbose=True,
                use_opus=True,
            )
            print(f"INFO|Translation saved to {trans_out}", flush=True)
            if phoneme_timestamps:
                _enrich_translated_source_phones(audiobase)

    # --- Stage 7: Publish best transcript for lyrics viewer playback ---
    _write_playback_transcript(audiobase, settings=settings, translation_mode=translation_mode)

    emit_progress(68, "Text pipeline complete")
    return str((Path.cwd() / audiobase).resolve()), (detectlang or lang_code)

def notesanalysis(af, output_root: Path, sr=48000, beat_strength_quantile=0.60, min_relative_beat_strength=1.05,
                  min_beat_gap_ms=120, beat_tolerance_ms=20):
    """Analyze pitch-class strength and beat timing; write .npz files for the lyrics viewer.

    Produces two compressed arrays per run:
        {audio_base}_novocs_analysis.npz  — instrumental / drum rhythm source
        {audio_base}_vocs_analysis.npz    — vocal stem

    Each .npz contains per-frame HPCP (12-bin chroma), beat maps, BPM, and metadata
    consumed by frase_core.LyricsSession.load_analysis_data().
    """
    import numpy as np
    import librosa

    def _prepare_madmom_numpy_aliases():
        # Older madmom versions use removed NumPy aliases (e.g., np.float).
        if not hasattr(np, "float"):
            np.float = float
        if not hasattr(np, "int"):
            np.int = int
        if not hasattr(np, "complex"):
            np.complex = np.complex128

    NOTE_NAMES = np.array(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    af_path = Path(af)
    audio_base = af_path.name

    def _score_beats(beats, bpm, confidence, duration_sec):
        """Heuristic quality score for a beat track (higher = more plausible)."""
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

    def extract_best_rhythm(rhythm_file_path, rhythm_audio, duration_sec, sr_local):
        """Beat tracking via madmom RNN+DBN, falling back to librosa HPSS beat_track."""
        try:
            _prepare_madmom_numpy_aliases()
            from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor

            fps = 100
            activations = np.asarray(
                RNNBeatProcessor(fps=fps)(str(rhythm_file_path)),
                dtype=np.float32,
            )
            beats = np.asarray(
                DBNBeatTrackingProcessor(fps=fps, min_bpm=55, max_bpm=215)(activations),
                dtype=np.float32,
            )

            if len(beats) >= 2:
                median_interval = float(np.median(np.diff(beats)))
                bpm = 60.0 / median_interval if median_interval > 0 else 0.0
            else:
                bpm = 0.0

            if len(beats) > 0 and len(activations) > 0:
                beat_indices = np.clip((beats * fps).astype(np.int64), 0, len(activations) - 1)
                confidence = float(np.mean(activations[beat_indices]))
            else:
                confidence = 0.0

            score = _score_beats(beats, bpm, confidence, duration_sec)
            if np.isfinite(score) and len(beats) >= 2 and bpm > 0:
                return {
                    "method": "madmom_dbn_rnn",
                    "bpm": float(bpm),
                    "beats": beats,
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "score": float(score),
                }

            print("INFO|madmom beat tracker produced weak output; falling back to librosa.", flush=True)
        except Exception as exc:
            print(f"INFO|madmom beat tracker unavailable; falling back to librosa ({exc})", flush=True)

        audio_np = np.asarray(rhythm_audio, dtype=np.float32)
        _, y_percussive = librosa.effects.hpss(audio_np)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr_local)
        beats = librosa.frames_to_time(beat_frames, sr=sr_local).astype(np.float32)

        tempo_arr = np.atleast_1d(tempo)
        bpm = float(tempo_arr[0]) if tempo_arr.size > 0 else 0.0
        if len(beats) > 0:
            beat_density = len(beats) / max(duration_sec, 1e-6)
            confidence = float(np.clip(beat_density / 4.0, 0.0, 1.0))
        else:
            confidence = 0.0

        return {
            "method": "librosa_beat_track",
            "bpm": bpm,
            "beats": beats,
            "confidence": confidence,
            "score": _score_beats(beats, bpm, confidence, duration_sec),
        }

    def detect_beats_from_energy_spikes(audio, sr_local, min_gap_ms_local):
        """Drum-only beat detection using drummervisual-style normalized RMS thresholding."""
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

    def _ensure_12_bin_matrix(values):
        """Normalize chroma/HPCP output to shape (num_frames, 12)."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return np.zeros((1, 12), dtype=np.float32)
        if arr.shape[1] == 12:
            return arr
        if arr.shape[0] == 12:
            return arr.T
        fixed = np.zeros((arr.shape[0], 12), dtype=np.float32)
        width = min(arr.shape[1], 12)
        fixed[:, :width] = arr[:, :width]
        return fixed

    def extract_note_strengths(audio_path, audio, sr_local):
        """Per-frame pitch-class strength via madmom DeepChroma, else librosa chroma_cqt."""
        duration_sec = float(len(np.asarray(audio, dtype=np.float32))) / float(sr_local)

        def _frame_timing(num_frames):
            if num_frames <= 0 or duration_sec <= 0:
                return np.zeros(0, dtype=np.float32), 512, (512 * 1000.0) / float(sr_local)
            frame_ms = (duration_sec * 1000.0) / float(num_frames)
            hop_length = max(1, int(round(sr_local * frame_ms / 1000.0)))
            frame_times = (
                np.arange(num_frames, dtype=np.float32)
                * np.float32(duration_sec / float(num_frames))
            )
            return frame_times, hop_length, frame_ms

        try:
            _prepare_madmom_numpy_aliases()
            from madmom.audio.chroma import DeepChromaProcessor

            fps = 100.0
            chroma = _ensure_12_bin_matrix(DeepChromaProcessor(fps=fps)(str(audio_path)))
            if chroma.shape[0] > 0:
                frame_times, hop_length, frame_ms = _frame_timing(chroma.shape[0])
                print(
                    f"INFO|madmom deep chroma frames={chroma.shape[0]} "
                    f"frame_ms={frame_ms:.2f} duration={duration_sec:.1f}s",
                    flush=True,
                )
                return chroma, frame_times, hop_length, "madmom_deepchroma", frame_ms

            print("INFO|madmom deep chroma returned no frames; falling back to librosa.", flush=True)
        except Exception as exc:
            print(f"INFO|madmom deep chroma unavailable; falling back to librosa ({exc})", flush=True)

        hop_length = 512
        chroma = librosa.feature.chroma_cqt(
            y=np.asarray(audio, dtype=np.float32),
            sr=sr_local,
            hop_length=hop_length,
        )
        chroma = _ensure_12_bin_matrix(chroma.T)
        if chroma.shape[0] == 0:
            chroma = np.zeros((1, 12), dtype=np.float32)
        frame_times = librosa.frames_to_time(
            np.arange(chroma.shape[0]),
            sr=sr_local,
            hop_length=hop_length,
        ).astype(np.float32)
        frame_ms = (hop_length * 1000.0) / float(sr_local)
        return chroma, frame_times, hop_length, "librosa_chroma_cqt", frame_ms

    def filter_beats_by_strength(audio, sr_local, beats):
        """Drop weak or too-close beats using quantile + relative envelope thresholds."""
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
        """Load audio, extract features, and write a compressed .npz analysis file."""
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        rhythm_source = rhythm_file_path or file_path
        rhythm_audio, _ = librosa.load(rhythm_source, sr=sr, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        rhythm_audio = np.asarray(rhythm_audio, dtype=np.float32)

        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            audio = audio / audio_max
        audio *= 20.0

        rhythm_max = np.max(np.abs(rhythm_audio))
        if rhythm_max > 0:
            rhythm_audio = rhythm_audio / rhythm_max
        rhythm_audio *= 20.0

        frame_size = 2048
        analysis_duration_sec = float(len(audio)) / float(sr)
        frame_hpcps, frame_times, hop_size, note_feature_method, frame_ms = extract_note_strengths(
            file_path, audio, sr
        )
        frame_hpcps = np.asarray(frame_hpcps, dtype=np.float32)
        if frame_hpcps.ndim != 2 or frame_hpcps.shape[0] == 0:
            frame_hpcps = np.zeros((1, 12), dtype=np.float32)
            frame_times = np.zeros(1, dtype=np.float32)
            frame_ms = analysis_duration_sec * 1000.0
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
            print("Drum stem detected - using RMS-threshold beat detection...", flush=True)
            filtered_beats, beat_strengths, beat_strength_threshold, bpm = \
                detect_beats_from_energy_spikes(rhythm_audio, sr, min_beat_gap_ms)
            beats = filtered_beats  # raw == filtered for npz consistency
            confidence = 1.0
            selected_method = "rms_visual"
        else:
            rhythm = extract_best_rhythm(rhythm_source, rhythm_audio, duration_sec, sr)
            bpm = rhythm["bpm"]
            beats = rhythm["beats"]
            filtered_beats, beat_strengths, beat_strength_threshold = filter_beats_by_strength(rhythm_audio, sr, beats)
            confidence = rhythm["confidence"]
            selected_method = rhythm["method"]

        print(
            f"INFO|Analysis backends notes={note_feature_method} rhythm={selected_method}",
            flush=True,
        )

        beat_map = np.zeros(num_frames, dtype=bool)
        beat_centers = np.zeros(num_frames, dtype=bool)
        beat_tolerance_sec = max(0.0, float(beat_tolerance_ms) / 1000.0)
        for beat_time in filtered_beats:
            if num_frames <= 0:
                break
            deltas = np.abs(frame_times - float(beat_time))
            frame_index = int(np.argmin(deltas))
            beat_centers[frame_index] = True
            within = np.where(deltas <= beat_tolerance_sec)[0]
            if len(within) == 0:
                beat_map[frame_index] = True
            else:
                beat_map[within] = True
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
            note_feature_method=np.array([note_feature_method]),
            beat_strengths_raw=beat_strengths,
            beat_strength_threshold=np.array([beat_strength_threshold]),
            beat_strength_quantile=np.array([beat_strength_quantile]),
            min_relative_beat_strength=np.array([min_relative_beat_strength]),
            min_beat_gap_ms=np.array([min_beat_gap_ms]),
            beat_tolerance_ms=np.array([beat_tolerance_ms]),
            sample_rate=np.array([sr]),
            frame_size=np.array([frame_size]),
            hop_size=np.array([hop_size]),
            frame_ms=np.array([frame_ms], dtype=np.float32),
            analysis_duration_sec=np.array([analysis_duration_sec], dtype=np.float32),
        )

    # --- Instrumental / drum rhythm analysis ---
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

    # --- Vocal stem analysis ---
    emit_progress(87, "Analyzing vocals notes/beats...")
    vocal_path = f"{af}/vocals.mp3"
    analyze_and_save(audio_base, vocal_path, "vocs_analysis")

    emit_progress(96, "Analysis files saved")


def main():
    """Worker entry point: parse CLI args, run splitter + notesanalysis, return exit code."""
    if len(sys.argv) < 2:
        print("ERROR|Missing input file path", flush=True)
        return 2

    file_path = sys.argv[1]
    lang_code = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    target_lang = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
    output_root_arg = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else None
    translation_mode = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "none"
    raw_settings = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "{}"

    output_root = Path(output_root_arg) if output_root_arg else default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    # All relative paths in splitter() resolve against the output directory.
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
            target_lang=target_lang,
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
    # Normally launched by Sonex.py QProcess, not invoked directly by the user.
    print("You've ran a worker process! This is not meant to be run directly. If you're seeing this, something went wrong.", flush=True)
    sys.exit(main())
