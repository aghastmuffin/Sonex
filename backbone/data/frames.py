# define nouns ("structs")
import librosa
import numpy as np

class AudioSignal:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate


class AudioContext:
    def __init__(self, path, sr=None):
        self.path = path
        self.y, self.sr = librosa.load(path, sr=sr, mono=True)
        self._mel_cache = {}

    def mel_spectrogram(
        self,
        *,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        power=2.0
    ):
        key = (n_mels, n_fft, hop_length, power)
        if key not in self._mel_cache:
            S = librosa.feature.melspectrogram(
                y=self.y,
                sr=self.sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=power,
            )
            self._mel_cache[key] = librosa.power_to_db(S, ref=np.max)
        return self._mel_cache[key]