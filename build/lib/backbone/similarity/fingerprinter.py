import librosa
import numpy as np

def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr
def chroma_fingerprint(y, sr):
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=512
    )
    # Aggregate over time
    mean = chroma.mean(axis=1)
    std = chroma.std(axis=1)

    fp = np.concatenate([mean, std])
    return fp.astype(np.float32)
def normalize(v):
    return v / np.linalg.norm(v)
def segment_audio(y, sr, segment_len=8.0, hop_len=4.0):
    seg_samples = int(segment_len * sr)
    hop_samples = int(hop_len * sr)

    segments = []
    for i in range(0, len(y) - seg_samples, hop_samples):
        segments.append(y[i:i+seg_samples])
    return segments
