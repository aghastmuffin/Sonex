
#Analysis Library
from essentia.standard import MonoLoader, FrameGenerator, Windowing, Spectrum, SpectralPeaks, HPCP, RhythmExtractor2013
import numpy as np
import threading

# Beat filtering/timing controls (increase strictness to reduce false drum hits)
BEAT_STRENGTH_QUANTILE = 0.60
MIN_RELATIVE_BEAT_STRENGTH = 1.05
MIN_BEAT_GAP_MS = 120
BEAT_TOLERANCE_MS = 20


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
if __name__ == "__main__":
    AUDIO = "tuchat"
    #file_path = f"{AUDIO}/{AUDIO}.mp3"
    file_path = f"{AUDIO}/htdemucs/{AUDIO}/no_vocals.mp3"
    file_path1 = f"{AUDIO}/vocals.mp3"
    sr = 48000 # 1ms = 48 samples
    t1 = threading.Thread(target=save_novocs, args=(AUDIO, file_path, sr)) #recommended usage
    t2 = threading.Thread(target=savevocs, args=(AUDIO, file_path1, sr))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    