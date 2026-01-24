import librosa
import numpy as np
from backbone.data.frames import AudioSignal, CQTFrame

def compute_cqt(audio: "AudioSignal", hop_length=512, bins_per_octave=12):
    C = librosa.cqt(
        audio.y,
        sr=audio.sr,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave
    )

    magnitudes = np.abs(C)
    freqs = librosa.cqt_frequencies(
        n_bins=magnitudes.shape[0],
        fmin=librosa.note_to_hz("C1"),
        bins_per_octave=bins_per_octave
    )
    times = librosa.frames_to_time(
        np.arange(magnitudes.shape[1]),
        sr=audio.sr,
        hop_length=hop_length
    )

    return CQTFrame(magnitudes, freqs, times, audio.sr, hop_length)

