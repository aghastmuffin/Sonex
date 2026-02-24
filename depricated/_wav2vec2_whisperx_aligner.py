import json, re, unicodedata, subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

def generate_aligned(
    head_folder,
    vocals="vocals.mp3",
    transcript="vocals_whisper_segments.json",
    device="cuda",
    compute_type="float16",
    language="en",
    align_model_name: Optional[str] = None,
    allow_fuzzy=False,
    fuzzy_max_lookahead=6,
    # extra robustness knobs:
    max_chars_per_segment=220,   # chunk long lines
    pad_window_s=0.35,           # give aligner a little slack
):
    HERE = Path(head_folder)
    AUDIO_MP3 = HERE / vocals
    WHISPER_JSON = HERE / transcript

    OUTDIR = HERE / "_whisperx_out"
    OUTDIR.mkdir(exist_ok=True)

    WAV = OUTDIR / "vocals.wav"
    OUT_JSON = HERE / "mfa_vocals_whisper_segments.json"

    def run(cmd):
        subprocess.run(cmd, check=True)

    _word_clean_re = re.compile(r"[^a-z0-9']+")
    def norm(w: str) -> str:
        w = (w or "").strip().lower().replace("’", "'")
        w = unicodedata.normalize("NFKD", w)
        w = w.encode("ascii", "ignore").decode("ascii")
        w = _word_clean_re.sub("", w)
        return w

    def build_fuzzy_mapping(whisper_norm_list, aligned_norm_list, max_lookahead):
        mapping = [None] * len(whisper_norm_list)
        i = 0
        j = 0
        while i < len(whisper_norm_list) and j < len(aligned_norm_list):
            if whisper_norm_list[i] == aligned_norm_list[j]:
                mapping[i] = j
                i += 1
                j += 1
                continue

            found = None
            for di in range(0, max_lookahead + 1):
                for dj in range(0, max_lookahead + 1):
                    wi = i + di
                    aj = j + dj
                    if wi < len(whisper_norm_list) and aj < len(aligned_norm_list):
                        if whisper_norm_list[wi] == aligned_norm_list[aj]:
                            found = (di, dj)
                            break
                if found:
                    break

            if found:
                di, dj = found
                for k in range(i, i + di):
                    mapping[k] = None
                i += di
                j += dj
                mapping[i] = j
                i += 1
                j += 1
                continue

            if i + 1 < len(whisper_norm_list) and whisper_norm_list[i + 1] == aligned_norm_list[j]:
                mapping[i] = None
                i += 1
                continue
            if j + 1 < len(aligned_norm_list) and whisper_norm_list[i] == aligned_norm_list[j + 1]:
                j += 1
                continue

            mapping[i] = None
            i += 1
            j += 1
        return mapping

    # 1) load whisper segments
    segments: List[Dict[str, Any]] = json.loads(WHISPER_JSON.read_text(encoding="utf-8"))

    whisper_words_raw = []
    for seg in segments:
        for w in seg.get("words", []):
            whisper_words_raw.append(w["word"])

    if not whisper_words_raw:
        raise RuntimeError("No words found in whisper JSON.")

    # 2) mp3 -> wav (16k mono)
    run(["ffmpeg", "-y", "-i", str(AUDIO_MP3), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(WAV)])

    # 3) whisperx align (API)
    import whisperx
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        if compute_type == "float16":
            compute_type = "int8"

    audio = whisperx.load_audio(str(WAV))

    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
        model_name=align_model_name
    )

    # Build rough segments from Whisper segments (CRITICAL CHANGE)
    rough_segments: List[Dict[str, Any]] = []
    for seg in segments:
        # prefer seg["text"] if present, else reconstruct from words
        text = (seg.get("text") or "").strip()
        if not text:
            ws = seg.get("words", [])
            text = " ".join((w.get("word", "") or "").strip() for w in ws).strip()

        if not text:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        # Give the aligner a reasonable window, even if Whisper timing is slightly off
        start = max(0.0, start - pad_window_s)
        if end > 0:
            end = end + pad_window_s

        # Chunk overly long segments (songs often create huge segments)
        if len(text) > max_chars_per_segment:
            parts = []
            cur = []
            cur_len = 0
            for tok in text.split():
                add = len(tok) + (1 if cur else 0)
                if cur_len + add > max_chars_per_segment:
                    parts.append(" ".join(cur))
                    cur = [tok]
                    cur_len = len(tok)
                else:
                    cur.append(tok)
                    cur_len += add
            if cur:
                parts.append(" ".join(cur))

            # distribute time window roughly across chunks
            if end > start:
                dur = end - start
                step = dur / len(parts)
            else:
                step = 0.0

            for i, p in enumerate(parts):
                s_i = start + i * step
                e_i = start + (i + 1) * step if step > 0 else end
                rough_segments.append({"start": s_i, "end": e_i, "text": p})
        else:
            rough_segments.append({"start": start, "end": end, "text": text})

    if not rough_segments:
        raise RuntimeError("No rough segments constructed for alignment (check your Whisper JSON).")

    aligned = whisperx.align(
        transcript=rough_segments,
        model=align_model,
        align_model_metadata=metadata,
        audio=audio,
        device=device,
        return_char_alignments=False,
    )

    aligned_segments = aligned.get("segments", [])
    if not aligned_segments:
        raise RuntimeError("WhisperX align() returned no segments.")

    # Collect aligned words across all aligned segments
    aligned_words = []
    for seg in aligned_segments:
        for w in seg.get("words", []):
            if w is None:
                continue
            s = w.get("start")
            e = w.get("end")
            if s is None or e is None:
                continue
            lab = w.get("word") or w.get("text") or ""
            aligned_words.append((float(s), float(e), str(lab)))

    if not aligned_words:
        raise RuntimeError(
            "WhisperX alignment returned no word timestamps.\n"
            "This usually means the transcript is too far from the audio OR the segments are too long.\n"
            "Try: smaller max_chars_per_segment (e.g. 120) and allow_fuzzy=True."
        )

    # 4) map whisper words -> aligned words
    whisper_norm = [norm(w) for w in whisper_words_raw]
    aligned_norm = [norm(lab) for _, _, lab in aligned_words]

    if whisper_norm != aligned_norm:
        if not allow_fuzzy:
            n = min(len(whisper_norm), len(aligned_norm))
            first_bad = next((i for i in range(n) if whisper_norm[i] != aligned_norm[i]), None)
            raise RuntimeError(
                "WhisperX word sequence != Whisper word sequence.\n"
                f"Whisper words: {len(whisper_norm)} | WhisperX words: {len(aligned_norm)}\n"
                f"First mismatch index: {first_bad}\n"
                "Fix: re-run with allow_fuzzy=True."
            )
        mapping = build_fuzzy_mapping(whisper_norm, aligned_norm, fuzzy_max_lookahead)
        if all(m is None for m in mapping):
            raise RuntimeError("Fuzzy alignment failed to match any words.")
    else:
        mapping = list(range(len(whisper_norm)))

    # 5) overwrite whisper timings
    k = 0
    for seg in segments:
        ws = seg.get("words", [])
        if not ws:
            continue
        for w in ws:
            idx = mapping[k]
            if idx is not None:
                s, e, _ = aligned_words[idx]
                w["start"] = s
                w["end"] = e
            k += 1
        seg["start"] = ws[0].get("start", seg.get("start"))
        seg["end"] = ws[-1].get("end", seg.get("end"))

    OUT_JSON.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)