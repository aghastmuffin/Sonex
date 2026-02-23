import json, re, subprocess, shutil
import unicodedata
from pathlib import Path
from typing import Optional

def generate_aligned(
    head_folder,
    vocals="vocals.mp3",
    transcript="vocals_whisper_segments.json",
    # whisperx knobs:
    device="cuda",                 # "cpu" if needed
    compute_type="float16",        # "int8" for CPU usually
    language="en",
    whisperx_align_model=None,     # e.g. "WAV2VEC2_ASR_LARGE_LV60K_960H" or leave None (auto)
    # your existing behavior:
    allow_fuzzy=False,
    fuzzy_max_lookahead=6,
):
    """
    Drop-in replacement for MFA version, but using WhisperX wav2vec2 CTC alignment.

    Input:  vocals mp3 + Whisper segments JSON with segments[*].words[*].word
    Output: writes head_folder / "mfa_vocals_whisper_segments.json" (same name as your old OUT_JSON)
    """

    HERE = Path(head_folder)
    AUDIO_MP3 = HERE / vocals
    WHISPER_JSON = HERE / transcript

    OUTDIR = HERE / "_whisperx_out"
    OUTDIR.mkdir(exist_ok=True)

    WAV = OUTDIR / "vocals.wav"
    OUT_JSON = HERE / "mfa_vocals_whisper_segments.json"  # keep same output name for drop-in compatibility

    def run(cmd):
        subprocess.run(cmd, check=True)

    # --------- normalization + fuzzy mapping (same as your MFA version) ----------
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

    # --------- 1) load whisper segments ----------
    segments = json.loads(WHISPER_JSON.read_text(encoding="utf-8"))

    whisper_words_raw = []
    for seg in segments:
        for w in seg.get("words", []):
            whisper_words_raw.append(w["word"])

    full_transcript = " ".join(w.strip() for w in whisper_words_raw).strip()
    if not full_transcript:
        raise RuntimeError("No words found in whisper JSON.")

    # --------- 2) convert mp3 -> wav (16k mono s16) ----------
    run(["ffmpeg", "-y", "-i", str(AUDIO_MP3), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(WAV)])

    # --------- 3) run whisperx aligner ----------
    # We intentionally avoid re-transcribing; we only align your provided transcript.
    #
    # whisperx CLI expects an audio file + options and produces JSON under output_dir.
    # The "result.json" format contains "segments" with "words" including timestamps (when alignment succeeds).
    #
    # NOTE: different whisperx versions name outputs slightly differently; we search for the newest JSON.
    python_exe = shutil.which("python") or "python"

    cmd = [
        python_exe, "-m", "whisperx",
        str(WAV),
        "--task", "align",
        "--output_dir", str(OUTDIR),
        "--language", language,
        "--device", device,
        "--compute_type", compute_type,
        "--text", full_transcript,
    ]
    if whisperx_align_model:
        cmd += ["--align_model", whisperx_align_model]

    run(cmd)

    # --------- 4) find whisperx-produced json ----------
    # Common names: "result.json", "<audio>.json", etc. We'll grab the most recent .json in OUTDIR.
    json_candidates = sorted(OUTDIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not json_candidates:
        raise RuntimeError(f"WhisperX produced no JSON in {OUTDIR}")

    aligned_json_path = json_candidates[0]
    aligned = json.loads(aligned_json_path.read_text(encoding="utf-8"))

    aligned_segments = aligned.get("segments", aligned if isinstance(aligned, list) else None)
    if aligned_segments is None:
        raise RuntimeError(f"Unexpected WhisperX JSON structure in {aligned_json_path.name}")

    # Collect aligned words (word-level timestamps)
    aligned_words = []
    for seg in aligned_segments:
        for w in seg.get("words", []):
            # WhisperX sometimes uses "word" or "text"
            ww = w.get("word", w.get("text", ""))
            if ww is None:
                continue
            s = w.get("start", None)
            e = w.get("end", None)
            if s is None or e is None:
                continue
            aligned_words.append((float(s), float(e), str(ww)))

    if not aligned_words:
        raise RuntimeError(
            "WhisperX alignment returned no word timestamps. "
            "This can happen if the transcript diverges too much from audio."
        )

    # --------- 5) map whisper words -> aligned words (strict or fuzzy) ----------
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
                "Fix: re-run with allow_fuzzy=True or clean transcript."
            )
        mapping = build_fuzzy_mapping(whisper_norm, aligned_norm, fuzzy_max_lookahead)
        if all(m is None for m in mapping):
            raise RuntimeError("Fuzzy alignment failed to match any words.")
    else:
        mapping = list(range(len(whisper_norm)))

    # --------- 6) overwrite whisper timings with aligned timings ----------
    k = 0
    for seg in segments:
        ws = seg.get("words", [])
        if not ws:
            continue
        for w in ws:
            idx = mapping[k]
            if idx is not None:
                s, e, _lab = aligned_words[idx]
                w["start"] = s
                w["end"] = e
            k += 1
        seg["start"] = ws[0].get("start", seg.get("start"))
        seg["end"] = ws[-1].get("end", seg.get("end"))

    OUT_JSON.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)


def install_lang(*args, **kwargs):
    # Not applicable for WhisperX alignment (no MFA model downloads).
    raise NotImplementedError("WhisperX uses wav2vec2 align models; no MFA model install step.")