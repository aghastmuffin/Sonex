import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio


# ----------------------------
# Text cleaning & tokenization
# ----------------------------

def normalize_lyrics(text: str) -> str:
    text = text.lower()
    # keep letters, space, and apostrophe for contractions
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def words_from_text(text: str) -> List[str]:
    return [w for w in text.split(" ") if w]

def chars_from_words(words: List[str]) -> List[str]:
    # CTC forced alignment here is character-based (letters + apostrophe + space separators).
    # We'll align a string like: "hello world"
    return list(" ".join(words))


# ----------------------------
# CTC alignment (trellis + backtrack)
# (Adapted from torchaudio tutorial logic)
# ----------------------------

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

def build_trellis(emission: torch.Tensor, tokens: List[int], blank_id: int) -> torch.Tensor:
    """
    emission: (T, V) log-prob
    tokens: length N
    trellis: (T+1, N+1)
    """
    T, V = emission.shape
    N = len(tokens)
    trellis = torch.full((T + 1, N + 1), -float("inf"), device=emission.device)
    trellis[0, 0] = 0.0

    # Initialize first column (all blanks)
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], dim=0)

    for t in range(T):
        # Either stay at same token (emit blank) or move to next token (emit that token)
        stay = trellis[t, 1:] + emission[t, blank_id]
        change = trellis[t, :-1] + emission[t, tokens]
        trellis[t + 1, 1:] = torch.maximum(stay, change)

    return trellis

def backtrack(trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int) -> List[Point]:
    """
    Return best path as a list of Points from end to start.
    """
    T = trellis.size(0) - 1
    N = trellis.size(1) - 1

    j = N
    # start from best time index for full token sequence
    t = torch.argmax(trellis[1:, j]).item() + 1

    path: List[Point] = []
    while j > 0:
        # Score if we stayed (blank)
        stay = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score if we changed (token)
        change = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        if change > stay:
            j -= 1
            path.append(Point(j, t - 1, float(emission[t - 1, tokens[j]].exp().item())))
        t -= 1

        if t == 0:
            break

    return path[::-1]

def merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    """
    Convert character path into segments per character.
    """
    segments: List[Segment] = []
    if not path:
        return segments

    i1 = 0
    while i1 < len(path):
        i2 = i1
        while i2 < len(path) and path[i2].token_index == path[i1].token_index:
            i2 += 1
        label = transcript[path[i1].token_index]
        score = float(np.mean([p.score for p in path[i1:i2]]))
        segments.append(Segment(label=label, start=path[i1].time_index, end=path[i2 - 1].time_index + 1, score=score))
        i1 = i2
    return segments

def segments_to_words(char_segments: List[Segment]) -> List[Segment]:
    """
    Group character segments into word segments separated by spaces.
    """
    words: List[Segment] = []
    current = []
    for seg in char_segments:
        if seg.label == " ":
            if current:
                label = "".join(s.label for s in current)
                start = current[0].start
                end = current[-1].end
                score = float(np.mean([s.score for s in current]))
                words.append(Segment(label=label, start=start, end=end, score=score))
                current = []
        else:
            current.append(seg)

    if current:
        label = "".join(s.label for s in current)
        start = current[0].start
        end = current[-1].end
        score = float(np.mean([s.score for s in current]))
        words.append(Segment(label=label, start=start, end=end, score=score))
    return words


# ----------------------------
# Main
# ----------------------------

def main(vocals_wav: str, lyrics_txt: str, out_json: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A strong, widely used CTC acoustic model (speech-trained, but works surprisingly well on clean vocals).
    # If you later swap in a singing-adapted CTC model, alignment gets even better.
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    labels = bundle.get_labels()
    blank_id = 0  # for this bundle, labels[0] is typically "<pad>" / blank

    # Load & resample audio
    wav, sr = torchaudio.load(vocals_wav)
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    target_sr = bundle.sample_rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    wav = wav.to(device)

    # Read & normalize lyrics
    lyrics_raw = Path(lyrics_txt).read_text(encoding="utf-8")
    lyrics_norm = normalize_lyrics(lyrics_raw)
    words = words_from_text(lyrics_norm)
    transcript = "".join(chars_from_words(words))  # e.g. "hello world"

    # Map transcript chars to model label indices
    label_to_id = {c: i for i, c in enumerate(labels)}
    # Many wav2vec2 label sets use '|' as space. We handle that mapping.
    space_label = "|" if "|" in label_to_id else " "
    tokens = []
    for ch in transcript:
        if ch == " ":
            tokens.append(label_to_id[space_label])
        else:
            if ch not in label_to_id:
                raise ValueError(f"Character '{ch}' not in model labels. Labels include: {labels}")
            tokens.append(label_to_id[ch])

    # Compute emissions
    with torch.inference_mode():
        emissions, _ = model(wav)
        emissions = torch.log_softmax(emissions, dim=-1)[0]  # (T, V)

    # Build trellis & backtrack
    trellis = build_trellis(emissions, tokens, blank_id)
    path = backtrack(trellis, emissions, tokens, blank_id)

    # Convert to character segments
    char_segs = merge_repeats(path, transcript)

    # Convert frame indices to time
    # wav2vec2 emissions are downsampled vs waveform; use ratio
    num_frames = emissions.size(0)
    audio_len_s = wav.size(1) / sr
    seconds_per_frame = audio_len_s / num_frames

    # Group into words
    word_segs = segments_to_words(char_segs)

    results = []
    for wseg in word_segs:
        results.append({
            "word": wseg.label,
            "start": wseg.start * seconds_per_frame,
            "end": wseg.end * seconds_per_frame,
            "score": wseg.score
        })

    Path(out_json).write_text(json.dumps({
        "audio": str(vocals_wav),
        "lyrics_normalized": lyrics_norm,
        "words": results
    }, indent=2), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Aligned {len(results)} words.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("vocals_wav", help="Path to vocals.wav (preferably a clean stem)")
    ap.add_argument("lyrics_txt", help="Path to lyrics.txt (known words)")
    ap.add_argument("--out", default="alignment.json", help="Output JSON path")
    args = ap.parse_args()
    main(args.vocals_wav, args.lyrics_txt, args.out)