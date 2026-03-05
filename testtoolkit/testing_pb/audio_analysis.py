# audio_analysis_fixed.py
import numpy as np
from essentia.standard import (
    MonoLoader, FrameGenerator, Windowing, Spectrum, HPCP,
    KeyExtractor, RhythmExtractor2013
)

def analyze_audio(file_path):
    """
    Analyze audio file for:
    - Notes present (HPCP / pitch class profile)
    - Key (tonal center + scale)
    - BPM / beat positions
    """

    # -----------------------
    # 1️⃣ Load audio
    # -----------------------
    loader = MonoLoader(filename=file_path, sampleRate=44100)
    audio = loader().astype(np.float32)
    sr = 44100

    # Normalize and scale audio to boost HPCP detection
    audio = audio / max(np.max(np.abs(audio)), 1e-6)  # normalize -1..1
    audio *= 20.0  # scaling factor (adjust if needed)

    # -----------------------
    # 2️⃣ HPCP (Notes Present)
    # -----------------------
    frame_size = 4096
    hop_size = 2048
    window = Windowing(type='hann')
    spectrum = Spectrum()
    hpcp_algo = HPCP(size=12, referenceFrequency=440.0,
                     minFrequency=40, maxFrequency=5000,
                     harmonics=8, bandPreset=False)

    hpcp_accum = np.zeros(12, dtype=np.float32)
    frame_count = 0

    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        if np.all(frame == 0):
            continue
        win_frame = window(frame)
        mag_spectrum = spectrum(win_frame)
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr).astype(np.float32)
        hpcp_frame = hpcp_algo(freqs, mag_spectrum)
        if np.max(hpcp_frame) == 0:
            continue
        hpcp_accum += hpcp_frame
        frame_count += 1

    if frame_count == 0 or np.max(hpcp_accum) == 0:
        hpcp_avg = np.zeros(12, dtype=np.float32)
    else:
        hpcp_avg = hpcp_accum / frame_count
        hpcp_avg /= np.max(hpcp_avg)

    pitch_classes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    notes_present = {note: float(hpcp_avg[i]) for i, note in enumerate(pitch_classes)}

    # -----------------------
    # 3️⃣ Key Detection
    # -----------------------
    key_extractor = KeyExtractor()
    key, scale, strength = key_extractor(audio)
    key_info = {'key': key, 'scale': scale, 'strength': strength}

    # -----------------------
    # 4️⃣ BPM / Beats
    # -----------------------
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)
    bpm_info = {'bpm': bpm, 'beats': beats, 'confidence': beats_confidence}

    # -----------------------
    # 5️⃣ Combine results
    # -----------------------
    analysis = {
        'notes_present': notes_present,
        'key': key_info,
        'bpm': bpm_info
    }

    return analysis


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    file_path = "tuchat/tuchat.mp3"  # Replace with your file
    results = analyze_audio(file_path)

    print("=== Notes Present (HPCP) ===")
    for note, strength in results['notes_present'].items():
        print(f"{note}: {strength:.2f}")

    print("\n=== Key Detection ===")
    print(f"{results['key']['key']} {results['key']['scale']} (strength: {results['key']['strength']:.2f})")

    print("\n=== BPM / Beats ===")
    print(f"BPM: {results['bpm']['bpm']:.2f}, Confidence: {results['bpm']['confidence']:.2f}")
    print(f"First 10 beats (s): {results['bpm']['beats'][:10]}")