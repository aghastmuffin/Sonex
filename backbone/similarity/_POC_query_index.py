import librosa
import numpy as np
import hnswlib
import pickle
from collections import Counter

# ------------- CONFIG ----------------
INDEX_PATH = "music_hnsw.bin"
META_PATH = "meta.pkl"

SR = 22050
SEGMENT_LEN = 8.0
HOP_LEN = 4.0
DIM = 24
TOP_K = 5
# ------------------------------------

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

# Load index
index = hnswlib.Index(space="cosine", dim=DIM)
index.load_index(INDEX_PATH)
index.set_ef(100)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

def query_track(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    segments = segment_audio(y, sr)

    hits = []

    for seg in segments:
        fp = chroma_fingerprint(seg, sr)
        labels, distances = index.knn_query(fp, k=TOP_K)
        for lbl, dist in zip(labels[0], distances[0]):
            fname, seg_id = metadata[lbl]
            hits.append((fname, dist))

    return hits

if __name__ == "__main__":
    import sys
    q = sys.argv[1]

    hits = query_track(q)
    counts = Counter(fname for fname, _ in hits)

    print("\nTop matches:")
    for fname, c in counts.most_common(10):
        print(f"{fname}: {c} hits")
