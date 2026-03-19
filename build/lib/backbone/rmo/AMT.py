import librosa
import numpy as np
from numpy.linalg import norm
from itertools import groupby
from backbone.data.frames import AudioContext


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def cosine(a, b):
    # Higher epsilon to prevent numerical noise
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-6)

def compress_output(chord_list):
    """Groups identical sequential chords (e.g., C:maj x4)."""
    if not chord_list:
        return []
    result = []
    for chord, group in groupby(chord_list):
        count = len(list(group))
        result.append(f"{chord} (x{count})")
    return result

# --------------------------------------------------
# Chord templates
# --------------------------------------------------

def build_chord_templates():
    notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    # Triads
    C_maj = np.array([1,0,0,0,1,0,0,1,0,0,0,0])
    C_min = np.array([1,0,0,1,0,0,0,1,0,0,0,0])
    C_dim = np.array([1,0,0,1,0,0,1,0,0,0,0,0])
    C_aug = np.array([1,0,0,0,1,0,0,0,1,0,0,0])
    # 7ths
    C_maj7 = np.array([1,0,0,0,1,0,0,1,0,0,0,1])
    C_min7 = np.array([1,0,0,1,0,0,0,1,0,0,1,0])
    C_dom7 = np.array([1,0,0,0,1,0,0,1,0,0,1,0])
    # Sus
    C_sus2 = np.array([1,0,1,0,0,0,0,1,0,0,0,0])
    C_sus4 = np.array([1,0,0,0,0,1,0,1,0,0,0,0])

    templates = {}
    for i, n in enumerate(notes):
        templates[f"{n}:maj"]   = np.roll(C_maj, i)
        templates[f"{n}:min"]   = np.roll(C_min, i)
        templates[f"{n}:dim"]   = np.roll(C_dim, i)
        templates[f"{n}:aug"]   = np.roll(C_aug, i)
        templates[f"{n}:maj7"]  = np.roll(C_maj7, i)
        templates[f"{n}:min7"]  = np.roll(C_min7, i)
        templates[f"{n}:dom7"]  = np.roll(C_dom7, i)
        templates[f"{n}:sus2"]  = np.roll(C_sus2, i)
        templates[f"{n}:sus4"]  = np.roll(C_sus4, i)

    # Normalize templates for cosine
    for k in templates:
        templates[k] = templates[k] / (norm(templates[k]) + 1e-6)
    return templates, notes

# --------------------------------------------------
# Key detection helpers (Krumhansl-Schmuckler profiles)
# --------------------------------------------------

_KRUMHANSL_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KRUMHANSL_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def detect_key(chroma_vec, notes):
    # chroma_vec: shape (12,)
    chroma_norm = chroma_vec / (np.linalg.norm(chroma_vec) + 1e-9)
    major_scores = [cosine(chroma_norm, np.roll(_KRUMHANSL_MAJOR, i)) for i in range(12)]
    minor_scores = [cosine(chroma_norm, np.roll(_KRUMHANSL_MINOR, i)) for i in range(12)]

    maj_root = int(np.argmax(major_scores))
    min_root = int(np.argmax(minor_scores))

    if major_scores[maj_root] >= minor_scores[min_root]:
        return f"{notes[maj_root]} major", major_scores[maj_root], minor_scores[min_root]
    else:
        return f"{notes[min_root]} minor", major_scores[maj_root], minor_scores[min_root]

# --------------------------------------------------
# Audio Preprocessing
# --------------------------------------------------

def preprocess_audio(audio_path):
    print(f"Loading {audio_path}...")
    y, sr = librosa.load(audio_path, mono=True)
    
    # 1. HPSS
    print("Separating vocals/instruments from drums... (LIBROSA NATIVE)")
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # 2. Chroma on harmonic only
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, fmin=librosa.note_to_hz('C1'))

    # 3. Beat tracking on percussive
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
    
    # 4. Sync chroma to beats
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)

    return chroma_sync

# --------------------------------------------------
# Probabilities & Viterbi
# --------------------------------------------------

def emission_logprobs(chroma, templates):
    chords = list(templates.keys())
    tmpl = np.stack([templates[c] for c in chords])

    T = chroma.shape[1]
    N = len(chords)
    E = np.zeros((T, N))

    for t in range(T):
        frame = chroma[:, t]
        # Normalize frame per time-step to reduce loudness bias
        frame = frame / (norm(frame) + 1e-6)
        for j in range(N):
            sim = cosine(frame, tmpl[j])
            # Use squared cosine to amplify differences
            E[t, j] = sim ** 2 if sim > 0 else 0

    # Clamp negatives to zero, normalize per frame to sum to 1
    E = np.maximum(E, 0.01)  # Prevent zero probabilities
    E = librosa.util.normalize(E, norm=1, axis=1)
    return np.log(E + 1e-8), chords

def transition_logprobs(chords, stay_prob=0.15, rel_prob=0.15, other_prob=0.70):
    """
    - stay_prob: P(chord_t = chord_{t-1}) - reduced to allow more transitions
    - rel_prob:  P(switching to 'relative' quality (maj<->min, maj7<->min7) on SAME root
    - other_prob: distributed over all *other* chords, allowing more flexibility
    """
    N = len(chords)
    T = np.zeros((N, N))

    # Base tiny probability for every transition
    eps = 1e-6
    T[:] = eps

    for i, c1 in enumerate(chords):
        parts = c1.split(":")
        root = parts[0]
        quality = parts[1]

        # Stay
        T[i, i] += stay_prob

        # Relative quality swap (maj<->min, maj7<->min7, dom7 transitions)
        rel_map = {
            "maj": "min",
            "min": "maj",
            "maj7": "min7",
            "min7": "maj7",
            "dom7": "maj7",
            "dim": "aug",
            "aug": "dim",
            "sus2": "sus4",
            "sus4": "sus2"
        }
        
        if quality in rel_map:
            rel_quality = rel_map[quality]
            rel = f"{root}:{rel_quality}"
            if rel in chords:
                j = chords.index(rel)
                T[i, j] += rel_prob

        # Distribute other_prob over remaining chords more evenly
        remaining = other_prob
        # Count non-self, non-relative transitions
        denom = N - 1 - (1 if quality in rel_map and f"{root}:{rel_map[quality]}" in chords else 0)
        if denom > 0:
            tiny = remaining / denom
            for k in range(N):
                if k == i:
                    continue
                rel_chord = f"{root}:{rel_map[quality]}" if quality in rel_map else None
                if rel_chord and k == chords.index(rel_chord):
                    continue
                T[i, k] += tiny

        # Normalize row explicitly
        T[i] = T[i] / (T[i].sum() + 1e-12)

    return np.log(T + 1e-12)

def viterbi_decode(E_log, T_log):
    T, N = E_log.shape
    dp = np.zeros((T, N))
    back = np.zeros((T, N), dtype=int)

    dp[0] = E_log[0]

    for t in range(1, T):
        scores = T_log + dp[t-1].reshape(-1, 1)
        best_prev = np.argmax(scores, axis=0)
        back[t] = best_prev
        dp[t] = scores[best_prev, np.arange(N)] + E_log[t]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(dp[-1])

    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]

    return path

# --------------------------------------------------
# Frame-wise analysis
# --------------------------------------------------

def analyze_frames(audio_path, note_threshold=0.2):
    chroma = preprocess_audio(audio_path)
    templates, notes = build_chord_templates()

    # Global key from mean chroma (context of the whole song)
    global_key, _, _ = detect_key(chroma.mean(axis=1), notes)

    E_log, chords = emission_logprobs(chroma, templates)
    T_log = transition_logprobs(chords)
    path = viterbi_decode(E_log, T_log)

    results = []
    for t, chord_idx in enumerate(path):
        frame_vec = chroma[:, t]
        # Active notes by threshold within frame (self-context)
        active_notes = [notes[i] for i, v in enumerate(frame_vec) if v >= note_threshold]

        # Local key for this frame (frame-level context)
        local_key, _, _ = detect_key(frame_vec, notes)

        results.append({
            "frame": int(t),
            "chord": chords[chord_idx],     # Viterbi-smoothed chord label
            "active_notes": active_notes,   # All notes present above threshold
            "local_key": local_key,         # Key guess for this frame alone
            "global_key": global_key        # Key guess for whole song
        })
    return results

def get_chords(audio_path):
    # Preserved for compatibility: returns only the chord labels per frame
    frames = analyze_frames(audio_path)
    return [f["chord"] for f in frames]

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    file_name = "ADV.mp3"
    
    try:
        print("--- Starting Analysis ---")
        frames = analyze_frames(file_name)
        
        print("\n--- Detected Progression (Cleaned) ---")
        compressed = compress_output([f["chord"] for f in frames])
        for i in range(0, len(compressed), 6):
            print(" | ".join(compressed[i:i+6]))

        print("\n--- Frame Details---")
        for f in frames:
            print(f"t={f['frame']:03d} | chord={f['chord']:<5} | active_notes={f['active_notes']} | local_key={f['local_key']} | global_key={f['global_key']}")
            
    except Exception as e:
        print(f"\nERROR: {e}")