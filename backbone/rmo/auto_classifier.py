#Identify sections in a song
import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from audio.audiocontext import AudioContext


# ----------------------------
# Feature extraction
# ----------------------------

def extract_features(y, sr):
    # Beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Chroma (core structural feature)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sync = librosa.util.sync(chroma, beats)

    # RMS energy (used later, not structural)
    rms = librosa.feature.rms(y=y)
    rms_sync = librosa.util.sync(rms, beats)

    return chroma_sync, rms_sync, beats


# ----------------------------
# Structural segmentation
# ----------------------------

def segment_structure(chroma_sync, n_sections=8):
    # Self-similarity matrix
    affinity = librosa.segment.recurrence_matrix(
        chroma_sync,
        metric="cosine",
        mode="affinity",
        sym=True
    )

    # Convert to distance matrix
    distance = 1 - affinity

    # Cluster beats into sections
    clustering = AgglomerativeClustering(
        n_clusters=n_sections,
        metric="precomputed",
        linkage="average"
    )
    labels = clustering.fit_predict(distance)

    return labels


# ----------------------------
# Section feature aggregation
# ----------------------------

def aggregate_sections(labels, chroma_sync, rms_sync):
    sections = {}

    for idx, label in enumerate(labels):
        sections.setdefault(label, []).append(idx)

    section_data = {}

    for label, indices in sections.items():
        chroma_mean = np.mean(chroma_sync[:, indices], axis=1)
        rms_mean = float(np.mean(rms_sync[:, indices]))
        duration = len(indices)

        section_data[label] = {
            "indices": indices,
            "chroma": chroma_mean,
            "energy": rms_mean,
            "duration": duration
        }

    return section_data


# ----------------------------
# Structural repetition scoring
# ----------------------------

def compute_repetition_scores(section_data):
    labels = list(section_data.keys())
    repetition_scores = {label: 0.0 for label in labels}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = section_data[labels[i]]["chroma"]
            b = section_data[labels[j]]["chroma"]

            sim = 1 - cosine(a, b)

            if sim > 0.85:  # similarity threshold
                repetition_scores[labels[i]] += sim
                repetition_scores[labels[j]] += sim

    return repetition_scores


# ----------------------------
# Prominence scoring (genre-agnostic)
# ----------------------------

def compute_prominence(section_data, repetition_scores):
    prominence = {}

    for label, data in section_data.items():
        prominence[label] = (
            0.6 * repetition_scores[label] +
            0.2 * data["energy"] +
            0.2 * data["duration"]
        )

    return prominence


# ----------------------------
# High-level API
# ----------------------------

def analyze_song(path, n_sections=8):
    y, sr = librosa.load(path, sr=22050)

    chroma_sync, rms_sync, beats = extract_features(y, sr)
    labels = segment_structure(chroma_sync, n_sections)

    section_data = aggregate_sections(labels, chroma_sync, rms_sync)
    repetition_scores = compute_repetition_scores(section_data)
    prominence = compute_prominence(section_data, repetition_scores)

    primary_section = max(prominence, key=prominence.get)

    return {
        "primary_section": primary_section,
        "sections": section_data,
        "repetition_scores": repetition_scores,
        "prominence_scores": prominence
    }
