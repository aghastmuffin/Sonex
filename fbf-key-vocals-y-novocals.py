
#Analysis Library
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

if __name__ == "__main__":
    AUDIO = "22"
    #file_path = f"{AUDIO}/{AUDIO}.mp3"
    file_path = f"{AUDIO}/htdemucs/{AUDIO}.mp3"
    sr = 48000 # 1ms = 48 samples
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
    rhythm = extract_best_rhythm(audio, duration_sec)
    bpm = rhythm["bpm"]
    beats = rhythm["beats"]
    confidence = rhythm["confidence"]
    selected_method = rhythm["method"]

    # Create a beat map aligned to HPCP frame indices with tolerance for perceptual timing
    beat_map = np.zeros(num_frames, dtype=bool)
    beat_centers = np.zeros(num_frames, dtype=bool)
    beat_tolerance_ms = 35
    beat_tolerance_frames = max(1, int(round((beat_tolerance_ms / 1000.0) * sr / hop_size)))
    for beat_time in beats:
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
                        beat_times=beats,
                        beat_confidence=np.array([confidence]),
                        rhythm_method=np.array([selected_method]),
                        beat_tolerance_ms=np.array([beat_tolerance_ms]),
                        sample_rate=np.array([sr]),
                        frame_size=np.array([frame_size]),
                        hop_size=np.array([hop_size]))
    
    #scan vocals now
    file_path = f"{AUDIO}/vocals.mp3"
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
    rhythm = extract_best_rhythm(audio, duration_sec)
    bpm = rhythm["bpm"]
    beats = rhythm["beats"]
    confidence = rhythm["confidence"]
    selected_method = rhythm["method"]

    # Create a beat map aligned to HPCP frame indices with tolerance for perceptual timing
    beat_map = np.zeros(num_frames, dtype=bool)
    beat_centers = np.zeros(num_frames, dtype=bool)
    beat_tolerance_ms = 35
    beat_tolerance_frames = max(1, int(round((beat_tolerance_ms / 1000.0) * sr / hop_size)))
    for beat_time in beats:
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
                        beat_times=beats,
                        beat_confidence=np.array([confidence]),
                        rhythm_method=np.array([selected_method]),
                        beat_tolerance_ms=np.array([beat_tolerance_ms]),
                        sample_rate=np.array([sr]),
                        frame_size=np.array([frame_size]),
                        hop_size=np.array([hop_size]))