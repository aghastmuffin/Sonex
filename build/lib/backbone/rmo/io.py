# sonex/audio/io.py
import librosa
from backbone.data.frames import AudioSignal

def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return AudioSignal(y, sr)
