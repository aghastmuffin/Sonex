import os
import sys
import traceback
import faulthandler

faulthandler.enable()

language_dict = {
    'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german', 'it': 'italian', 'pt': 'portuguese',
    'ru': 'russian', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean', 'ar': 'arabic', 'hi': 'hindi',
    'bn': 'bengali', 'pa': 'punjabi', 'tr': 'turkish', 'vi': 'vietnamese', 'pl': 'polish', 'nl': 'dutch',
    'sv': 'swedish', 'no': 'norwegian', 'da': 'danish', 'fi': 'finnish', 'he': 'hebrew', 'el': 'greek',
    'th': 'thai', 'id': 'indonesian', 'uk': 'ukrainian', 'cs': 'czech', 'ro': 'romanian', 'hu': 'hungarian'
}


def emit_progress(value, label):
    print(f"PROGRESS|{int(value)}|{label}", flush=True)


def emit_demucs_active(is_active):
    print(f"DEMUCS_ACTIVE|{1 if is_active else 0}", flush=True)


def emit_demucs_progress(value, label="Demucs separating stems..."):
    print(f"DEMUCS_PROGRESS|{int(value)}|{label}", flush=True)


def splitter(file_path, lang_code=None, translation_mode="argos"):
    from backbone.ltra.letra_toolkit import transcribe, align, separate
    from backbone.ltra.argos_translate import translate_file

    translation_mode = (translation_mode or "argos").strip().lower()
    if translation_mode not in {"argos", "whisper", "both"}:
        translation_mode = "argos"

    emit_progress(10, "Separating stems...")
    emit_demucs_active(True)
    emit_demucs_progress(0, "Demucs separating stems...")
    try:
        separate(file_path, demucs_progress_cb=lambda value: emit_demucs_progress(value))
        emit_demucs_progress(100, "Demucs complete")
    finally:
        emit_demucs_active(False)

    audiobase = os.path.basename(file_path).removesuffix(".mp3")
    os.makedirs(audiobase, exist_ok=True)

    emit_progress(25, "Transcribing vocals...")
    _, detectlang = transcribe(
        f"{audiobase}/vocals.mp3",
        5,
        2,
        f"{audiobase}/vocals_whisper_segments.json",
        language=lang_code,
        task="transcribe",
    )

    emit_progress(40, "Aligning words...")
    align(
        f"{audiobase}/vocals.mp3",
        f"{audiobase}/vocals_whisper_segments.json",
        f"{audiobase}/lyrics.txt",
        language=lang_code,
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
                dictionary=f"{language_dict[mfa_lang]}_mfa",
                allow_fuzzy=True,
                fuzzy_max_lookahead=8,
            )
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
    
    source_lang = normalize_lang_code(detectlang or lang_code or "es")
    target_lang = normalize_lang_code("en")

    if translation_mode in {"whisper", "both"}:
        emit_progress(58, "Whisper translation pass...")
        transcribe(
            f"{audiobase}/vocals.mp3",
            5,
            2,
            f"{audiobase}/whisper_translated.json",
            language=(detectlang or lang_code),
            task="translate",
            reuse_existing=False,
        )

    if translation_mode in {"argos", "both"}:
        if source_lang == target_lang:
            print(f"Skipping Argos translation: source and target are both '{target_lang}'.", flush=True)
        else:
            emit_progress(62, "Argos translation pass...")
            translate_file(
                f"{audiobase}/vocals_whisper_segments.json",
                output_path=f"{audiobase}/argos_translated.json",
                from_lang=source_lang,
                to_lang=target_lang,
                verbose=True,
            )

    emit_progress(68, "Text pipeline complete")
    return audiobase, (detectlang or lang_code)


def notesanalysis(af, sr=48000, beat_strength_quantile=0.60, min_relative_beat_strength=1.05,
                  min_beat_gap_ms=120, beat_tolerance_ms=20):
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

    def extract_best_rhythm(audio, duration_sec):
        best = None
        for method in ("multifeature", "degara"):
            extractor = RhythmExtractor2013(method=method)
            bpm, beats, confidence, estimates, bpm_intervals = extractor(audio)
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

    def analyze_and_save(audio_name, file_path, out_name):
        audio = MonoLoader(filename=file_path, sampleRate=sr)().astype(np.float32)

        audio_max = np.max(np.abs(audio))
        if audio_max > 0:
            audio = audio / audio_max
        audio *= 20.0

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
        num_frames = len(frame_hpcps)
        duration_sec = num_frames * hop_size / sr

        rhythm = extract_best_rhythm(audio, duration_sec)
        bpm = rhythm["bpm"]
        beats = rhythm["beats"]
        filtered_beats, beat_strengths, beat_strength_threshold = filter_beats_by_strength(audio, sr, beats)
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

        np.savez_compressed(
            f"{audio_name}/{audio_name}_{out_name}.npz",
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
            beat_strength_quantile=np.array([beat_strength_quantile]),
            min_relative_beat_strength=np.array([min_relative_beat_strength]),
            min_beat_gap_ms=np.array([min_beat_gap_ms]),
            beat_tolerance_ms=np.array([beat_tolerance_ms]),
            sample_rate=np.array([sr]),
            frame_size=np.array([frame_size]),
            hop_size=np.array([hop_size]),
        )

    emit_progress(78, "Analyzing instrumental notes/beats...")
    novocal_path = f"{af}/htdemucs/{af}/no_vocals.mp3"
    analyze_and_save(af, novocal_path, "novocs_analysis")

    emit_progress(87, "Analyzing vocals notes/beats...")
    vocal_path = f"{af}/vocals.mp3"
    analyze_and_save(af, vocal_path, "vocs_analysis")

    emit_progress(96, "Analysis files saved")


def main():
    if len(sys.argv) < 2:
        print("ERROR|Missing input file path", flush=True)
        return 2

    file_path = sys.argv[1]
    lang_code = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    translation_mode = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "argos"

    try:
        emit_progress(5, "Starting...")
        audiobase, detected_lang = splitter(file_path, lang_code=lang_code, translation_mode=translation_mode)
        print(f"AUDIOBASE|{audiobase}", flush=True)
        if detected_lang:
            print(f"LANG|{detected_lang}", flush=True)
        emit_progress(70, "Audio prep complete")
        notesanalysis(audiobase)
        emit_progress(100, "Done")
        return 0
    except Exception as exc:
        print(f"ERROR|{exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
