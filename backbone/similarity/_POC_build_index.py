import os
import librosa
import numpy as np
import hnswlib
from tqdm import tqdm
import pickle

# ---------------- CONFIG ----------------
AUDIO_DIR = "audio"        # folder of .wav/.mp3
INDEX_PATH = "music_hnsw.bin"
META_PATH = "meta.pkl"

SR = 22050
SEGMENT_LEN = 8.0          # seconds
HOP_LEN = 4.0              # seconds
DIM = 24                   # chroma mean + std
MAX_ELEMENTS = 500_000

# ----------------------------------------

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def chroma_fingerprint(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    mean = chroma.mean(axis=1)
    std = chroma.std(axis=1)
    return normalize(np.concatenate([mean, std]).astype(np.float32))

def segment_audio(y, sr):
    seg = int(SEGMENT_LEN * sr)
    hop = int(HOP_LEN * sr)
    return [y[i:i+seg] for i in range(0, len(y) - seg, hop)]

# Create index
index = hnswlib.Index(space="cosine", dim=DIM)
index.init_index(
    max_elements=MAX_ELEMENTS,
    ef_construction=200,
    M=16
)

metadata = []
label = 0

for fname in tqdm(os.listdir(AUDIO_DIR)):
    if not fname.lower().endswith((".wav", ".mp3", ".flac")):
        continue

    path = os.path.join(AUDIO_DIR, fname)
    y, sr = librosa.load(path, sr=SR, mono=True)

    for i, seg in enumerate(segment_audio(y, sr)):
        fp = chroma_fingerprint(seg, sr)
        index.add_items(fp, label)
        metadata.append((fname, i))
        label += 1

print(f"Indexed {label} segments")

index.save_index(INDEX_PATH)
with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("Index + metadata saved")
