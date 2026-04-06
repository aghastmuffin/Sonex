import argparse
import copy
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from praatio import tgio
from typing import Optional


ACOUSTIC_MODELS = {
    "english": "english_mfa",
    "english_us": "english_us_arpa",
    "english_uk": "english_uk_mfa",
    "english_india": "english_india_mfa",
    "english_nigeria": "english_nigeria_mfa",

    "spanish": "spanish_mfa",
    "spanish_spain": "spanish_spain_mfa",
    "spanish_latin_america": "spanish_latin_america_mfa",

    "french": "french_mfa",
    "german": "german_mfa",
    "portuguese": "portuguese_mfa",
    "italian": "italian_mfa",
    "dutch": "dutch_mfa",
    "catalan": "catalan_mfa",

    "mandarin": "mandarin_mfa",
    "japanese": "japanese_mfa",
    "korean": "korean_mfa",

    "russian": "russian_mfa",
    "ukrainian": "ukrainian_mfa",
    "polish": "polish_mfa",
    "czech": "czech_mfa",
    "slovak": "slovak_mfa",
    "croatian": "croatian_mfa",
    "serbocroatian": "serbocroatian_mfa",
    "bulgarian": "bulgarian_mfa",
    "belarusian": "belarusian_mfa",

    "turkish": "turkish_mfa",
    "vietnamese": "vietnamese_mfa",
    "thai": "thai_mfa",
}

DICTIONARY_MODELS = {
    "english": "english_us_arpa",
    "english_us": "english_us_arpa",
    "english_uk": "english_uk_mfa",
    "english_india": "english_india_mfa",
    "english_nigeria": "english_nigeria_mfa",
    "english_us_arpa": "english_us_arpa",

    "spanish": "spanish_mfa",
    "spanish_spain": "spanish_spain_mfa",
    "spanish_latin_america": "spanish_latin_america_mfa",

    "french": "french_mfa",
    "german": "german_mfa",
    "portuguese": "portuguese_mfa",
    "italian": "italian_mfa",
    "dutch": "dutch_mfa",
    "catalan": "catalan_mfa",

    "mandarin": "mandarin_mfa",
    "japanese": "japanese_mfa",
    "korean": "korean_mfa",

    "russian": "russian_mfa",
    "ukrainian": "ukrainian_mfa",
    "polish": "polish_mfa",
    "czech": "czech_mfa",
    "slovak": "slovak_mfa",
    "croatian": "croatian_mfa",
    "serbocroatian": "serbocroatian_mfa",
    "bulgarian": "bulgarian_mfa",
    "belarusian": "belarusian_mfa",

    "turkish": "turkish_mfa",
    "vietnamese": "vietnamese_mfa",
    "thai": "thai_mfa",
}


PHONEMIZER_LANGUAGE_MAP = {
    "english_us": "en-us",
    "english_uk": "en-gb",
    "english_india": "en-in",
    "english_nigeria": "en-us",
    "english": "en-us",
    "spanish_latin_america": "es",
    "spanish_spain": "es",
    "spanish": "es",
    "french": "fr-fr",
    "german": "de",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "catalan": "ca",
    "mandarin": "cmn",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
    "ukrainian": "uk",
    "polish": "pl",
    "czech": "cs",
    "slovak": "sk",
    "croatian": "hr",
    "serbocroatian": "hr",
    "bulgarian": "bg",
    "belarusian": "be",
    "turkish": "tr",
    "vietnamese": "vi",
    "thai": "th",
}


def _looks_like_word_tier(name: str) -> bool:
    n = name.strip().lower()
    return n == "word" or n == "words" or n.endswith(" - words") or "words" in n or "word" in n


def _looks_like_phone_tier(name: str) -> bool:
    n = name.strip().lower()
    return n == "phone" or n == "phones" or n.endswith(" - phones") or "phones" in n or "phone" in n


def _read_phone_intervals_from_textgrid(textgrid_path, include_silence: bool = False):
    tg = tgio.openTextgrid(str(textgrid_path), False)
    phone_tier_name = next((n for n in tg.tierNameList if _looks_like_phone_tier(n)), None)
    if not phone_tier_name:
        raise RuntimeError(f"Could not find a phone tier in {textgrid_path}. Tiers: {tg.tierNameList}")

    phone_tier = tg.tierDict[phone_tier_name]
    phone_intervals = [(float(s), float(e), str(lab).strip()) for (s, e, lab) in phone_tier.entryList]
    if include_silence:
        return phone_intervals

    return [
        (s, e, lab)
        for (s, e, lab) in phone_intervals
        if lab and lab.lower() not in {"sp", "sil"}
    ]


def export_phone_timestamps_from_textgrid(
    textgrid_path,
    out_json: Optional[str] = None,
    include_silence: bool = False,
):
    """
    Export a flat phone-level timestamp JSON from an MFA TextGrid.

    Output schema:
    [
      {"phone": "AH0", "start": 0.31, "end": 0.39},
      ...
    ]
    """
    tg_path = Path(textgrid_path)
    if not tg_path.exists():
        raise RuntimeError(f"Missing TextGrid: {tg_path}")

    intervals = _read_phone_intervals_from_textgrid(tg_path, include_silence=include_silence)
    payload = [{"phone": lab, "start": s, "end": e} for s, e, lab in intervals]

    out_path = Path(out_json) if out_json else tg_path.with_name(f"{tg_path.stem}_phones.json")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    return payload


def _infer_phonemizer_language(*candidates: Optional[str]) -> str:
    lowered = [str(c or "").lower() for c in candidates]
    for key in sorted(PHONEMIZER_LANGUAGE_MAP.keys(), key=len, reverse=True):
        if any(key in c for c in lowered):
            return PHONEMIZER_LANGUAGE_MAP[key]
    return "en-us"


def _normalize_word_for_phonemizer(word_text: str) -> str:
    txt = str(word_text or "").strip().replace("’", "'")
    txt = re.sub(r"[^\w']+", "", txt, flags=re.UNICODE)
    txt = txt.replace("_", "")
    if txt.isdigit():
        return ""
    return txt.lower()


def _fallback_phone_labels(word_text: str):
    txt = _normalize_word_for_phonemizer(word_text)
    if not txt:
        return ["spn"]
    return [ch for ch in txt if ch.strip()]


def _distribute_labels_over_word(labels, start: float, end: float):
    labels = [str(lab).strip() for lab in labels if str(lab).strip()]
    if not labels:
        labels = ["spn"]

    s = float(start)
    e = float(end)
    if e <= s:
        e = s + 0.001

    dur = e - s
    n = len(labels)
    out = []
    for i, lab in enumerate(labels):
        ps = s + (dur * i / n)
        pe = s + (dur * (i + 1) / n)
        out.append({
            "phone": lab,
            "start": ps,
            "end": pe,
        })
    return out


def _split_text_into_phone_segments(word_text: str, n_segments: int):
    text = str(word_text or "").strip()
    if n_segments <= 0:
        return []
    if not text:
        return [{"text": "", "char_start": 0, "char_end": 0} for _ in range(n_segments)]

    chars = list(text)
    total = len(chars)
    if total == 0:
        return [{"text": "", "char_start": 0, "char_end": 0} for _ in range(n_segments)]

    if n_segments <= total:
        base = total // n_segments
        rem = total % n_segments
        lengths = [base + (1 if i < rem else 0) for i in range(n_segments)]
    else:
        lengths = [1] * total + [0] * (n_segments - total)

    out = []
    cursor = 0
    for length in lengths:
        if length > 0:
            start = cursor
            end = cursor + length
            chunk = "".join(chars[start:end])
            cursor = end
        else:
            start = total - 1
            end = total
            chunk = chars[-1]
        out.append({
            "text": chunk,
            "char_start": start,
            "char_end": end,
        })

    return out


def _build_phonemizer_word_runner(language: str, enabled: bool = True):
    if not enabled:
        return None

    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
    except Exception:
        print("Info: phonemizer is not installed; falling back to character-based phone splits.")
        return None

    separator = Separator(phone=" ", syllable="", word=" | ")
    state = {"enabled": True}
    cache = {}

    def run_for_word(word_text: str):
        token = _normalize_word_for_phonemizer(word_text)
        if not token:
            return []
        if token in cache:
            return list(cache[token])
        if not state["enabled"]:
            return []

        try:
            raw = phonemize(
                token,
                language=language,
                backend="espeak",
                separator=separator,
                strip=True,
                preserve_punctuation=False,
                njobs=1,
            )
        except Exception as exc:
            state["enabled"] = False
            print(f"Info: phonemizer disabled after backend error: {exc}")
            cache[token] = []
            return []

        if isinstance(raw, list):
            raw = raw[0] if raw else ""

        phones = [p for p in re.split(r"\s+", str(raw).replace("|", " ").strip()) if p]
        cache[token] = phones
        return list(phones)

    return run_for_word


def _enrich_words_with_phone_segments(
    segments,
    phonemizer_language: Optional[str] = None,
    use_phonemizer: bool = True,
):
    runner = _build_phonemizer_word_runner(
        language=phonemizer_language or "en-us",
        enabled=bool(use_phonemizer),
    )

    stats = {
        "words_total": 0,
        "words_filled_fallback": 0,
        "words_segmented": 0,
        "phonemizer_enabled": bool(runner),
        "phonemizer_language": phonemizer_language or "en-us",
    }

    for seg in segments:
        for w in seg.get("words", []):
            stats["words_total"] += 1

            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            word_text = str(w.get("word", "")).strip()

            clean_existing = []
            for p in w.get("phones", []) or []:
                if "start" not in p or "end" not in p:
                    continue
                ps = float(p["start"])
                pe = float(p["end"])
                if pe <= ps:
                    continue
                label = str(p.get("phone", "")).strip() or "spn"
                clean_existing.append({"phone": label, "start": ps, "end": pe})

            if not clean_existing:
                labels = runner(word_text) if runner else []
                if not labels:
                    labels = _fallback_phone_labels(word_text)
                clean_existing = _distribute_labels_over_word(labels, ws, we)
                stats["words_filled_fallback"] += 1

            w["phones"] = clean_existing

            text_slices = _split_text_into_phone_segments(word_text, len(clean_existing))
            phone_segments = []
            for p, t in zip(clean_existing, text_slices):
                phone_segments.append({
                    "text": t["text"],
                    "char_start": int(t["char_start"]),
                    "char_end": int(t["char_end"]),
                    "phone": str(p.get("phone", "")),
                    "start": float(p["start"]),
                    "end": float(p["end"]),
                })

            w["phone_segments"] = phone_segments
            stats["words_segmented"] += 1

    return stats

#HERE = Path(__file__).resolve().parent
def generate_aligned(head_folder, vocals="vocals.mp3", transcript="vocals_whisper_segments.json", acoustic="english_us_arpa", dictionary="english_us_arpa", allow_fuzzy=False, fuzzy_max_lookahead=6):
    #DEPRECATED
    print("Deprecation Notice. This function should not be used.")
    # Ensure path arithmetic works even when head_folder is a string.
    HERE = Path(head_folder)
    AUDIO_MP3 = HERE / vocals
    WHISPER_JSON = HERE / transcript

    CORPUS = HERE / "_mfa_corpus"
    OUTDIR = HERE / "_mfa_out"
    CORPUS.mkdir(exist_ok=True)
    OUTDIR.mkdir(exist_ok=True)

    WAV = CORPUS / "vocals.wav"
    TXT = CORPUS / "vocals.txt"
    TEXTGRID = OUTDIR / "vocals.TextGrid"
    OUT_JSON = HERE / "mfa_vocals_whisper_segments.json"

    def run(cmd):
        subprocess.run(cmd, check=True)

    _word_clean_re = re.compile(r"[^a-z0-9']+")
    def norm(w: str) -> str:
        w = w.strip().lower().replace("’", "'")
        w = unicodedata.normalize("NFKD", w)
        w = w.encode("ascii", "ignore").decode("ascii")
        w = _word_clean_re.sub("", w)
        return w

    def build_fuzzy_mapping(whisper_norm_list, mfa_norm_list, max_lookahead):
        mapping = [None] * len(whisper_norm_list)
        i = 0
        j = 0
        while i < len(whisper_norm_list) and j < len(mfa_norm_list):
            if whisper_norm_list[i] == mfa_norm_list[j]:
                mapping[i] = j
                i += 1
                j += 1
                continue

            # Look ahead for the next matching pair within a small window.
            found = None
            for di in range(0, max_lookahead + 1):
                for dj in range(0, max_lookahead + 1):
                    wi = i + di
                    mj = j + dj
                    if wi < len(whisper_norm_list) and mj < len(mfa_norm_list):
                        if whisper_norm_list[wi] == mfa_norm_list[mj]:
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

            # Small local fixes for one-off insert/delete.
            if i + 1 < len(whisper_norm_list) and whisper_norm_list[i + 1] == mfa_norm_list[j]:
                mapping[i] = None
                i += 1
                continue
            if j + 1 < len(mfa_norm_list) and whisper_norm_list[i] == mfa_norm_list[j + 1]:
                j += 1
                continue

            # Give up on this word and move both pointers.
            mapping[i] = None
            i += 1
            j += 1
        return mapping

    # 1) load whisper segments
    segments = json.loads(WHISPER_JSON.read_text(encoding="utf-8"))

    # 2) build transcript from whisper words (best chance of matching)
    whisper_words_raw = []
    for seg in segments:
        for w in seg.get("words", []):
            whisper_words_raw.append(w["word"])

    transcript = " ".join(w.strip() for w in whisper_words_raw).strip()
    if not transcript:
        raise RuntimeError("No words found in whisper JSON.")

    # 3) convert to basic wav (recommended by MFA docs)
    run(["ffmpeg", "-y", "-i", str(AUDIO_MP3), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(WAV)])

    # 4) write transcript file (same stem as audio)
    TXT.write_text(transcript + "\n", encoding="utf-8")

    #conda run -n mfa mfa instead of mfa
    run(["conda", "run", "-n", "mfa", "mfa", "align", str(CORPUS), dictionary, acoustic, str(OUTDIR), "--clean", "--overwrite"])

    if not TEXTGRID.exists():
        raise RuntimeError(f"Missing TextGrid: {TEXTGRID}")

    # 6) parse TextGrid, find a word tier
    # Support older praatio versions that don't accept includeEmptyIntervals kwarg.
    tg = tgio.openTextgrid(str(TEXTGRID), False)
    tier_names = tg.tierNameList

    # MFA can output tiers like "words", "Word", or per-speaker variants.
    def looks_like_word_tier(name: str) -> bool:
        n = name.strip().lower()
        return "word" == n or "words" == n or n.endswith(" - words") or "words" in n or "word" in n

    word_tier_name = next((n for n in tier_names if looks_like_word_tier(n)), None)
    if not word_tier_name:
        raise RuntimeError(f"Could not find a word tier. Tiers were: {tier_names}")

    word_tier = tg.tierDict[word_tier_name]
    intervals = [(float(s), float(e), lab.strip()) for (s, e, lab) in word_tier.entryList]
    # drop silence-like labels if present
    intervals = [(s,e,lab) for (s,e,lab) in intervals if lab and lab.lower() not in {"sp","sil"}]

    # 7) strict 1:1 word matching (fast + clean). If mismatch -> you need fuzzy matching.
    whisper_norm = [norm(w) for w in whisper_words_raw]
    mfa_norm = [norm(lab) for _, _, lab in intervals]

    if whisper_norm != mfa_norm:
        if not allow_fuzzy:
            # help debug without dumping everything
            n = min(len(whisper_norm), len(mfa_norm))
            first_bad = next((i for i in range(n) if whisper_norm[i] != mfa_norm[i]), None)
            raise RuntimeError(
                "MFA word sequence != Whisper word sequence.\n"
                f"Whisper words: {len(whisper_norm)} | MFA words: {len(mfa_norm)}\n"
                f"First mismatch index: {first_bad}\n"
                "Fix: make transcript match audio better OR re-run with allow_fuzzy=True."
            )
        mapping = build_fuzzy_mapping(whisper_norm, mfa_norm, fuzzy_max_lookahead)
        if all(m is None for m in mapping):
            raise RuntimeError(
                "Fuzzy alignment failed to match any words. "
                "Check transcript, dictionary, and language models."
            )
    else:
        mapping = list(range(len(whisper_norm)))

    # 8) overwrite whisper timings with MFA timings (same format)
    k = 0
    for seg in segments:
        ws = seg.get("words", [])
        if not ws:
            continue
        for w in ws:
            mfa_index = mapping[k]
            if mfa_index is not None:
                s, e, _lab = intervals[mfa_index]
                w["start"] = s
                w["end"] = e
            k += 1
        seg["start"] = ws[0]["start"]
        seg["end"] = ws[-1]["end"]

    OUT_JSON.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)
def install_lang(model_general, dictionary: Optional[str] = None):
    acoustic_model = ACOUSTIC_MODELS.get(model_general, model_general)
    dictionary_model = DICTIONARY_MODELS.get(dictionary, dictionary) if dictionary else DICTIONARY_MODELS.get(model_general, model_general)

    def run_mfa(args):
        subprocess.run(["conda", "run", "-n", "mfa", "mfa", *args], check=True)

    if not Path(acoustic_model).exists():
        run_mfa(["model", "download", "acoustic", acoustic_model])

    if dictionary_model and not Path(dictionary_model).exists():
        run_mfa(["model", "download", "dictionary", dictionary_model])

def generate_aligned_v2(
    head_folder,
    vocals="vocals.mp3",
    transcript="vocals_whisper_segments.json",
    acoustic="english_us_arpa",
    dictionary="english_us_arpa",
    allow_fuzzy=False,
    fuzzy_max_lookahead=6,
    # robustness knobs (safe defaults)
    force_fresh_run=True,          # nuke old corpus/out to avoid “random fixes”
    num_jobs=1,                    # determinism; bump if you want speed
    oov_token="spn",               # fallback token if dictionary supports it (common)
    drop_weird_tokens=True,        # filter junk from whisper words
    max_chars_per_chunk=220,       # if transcript is huge, chunking helps MFA stability
    chunk_silence_gap_s=0.35,      # small silence between chunks
    export_flat_phone_json=True,
    phonemizer_language: Optional[str] = None,
    use_phonemizer=True,
):
    """
    More robust MFA aligner:
    - normalizes + filters Whisper tokens BEFORE transcript creation
    - optional OOV handling: replaces unknown words w/ `oov_token` (if present in dict)
    - chunks long transcripts into multiple utterances to prevent catastrophic drift
    - forces fresh runs by deleting corpus/outdir to remove nondeterministic “random fixes”
    - optionally runs with num_jobs=1 for determinism

    Output files:
    - mfa_vocals_whisper_segments.json (word-level timings)
    - mfa_vocals_phone_segments.json (word + nested phone timings)
    - each word gets `phone_segments` with text slices mapped to phone timestamps
    """

    HERE = Path(head_folder)
    AUDIO_MP3 = HERE / vocals
    WHISPER_JSON = HERE / transcript

    CORPUS = HERE / "_mfa_corpus"
    OUTDIR = HERE / "_mfa_out"
    WAV = CORPUS / "vocals.wav"
    OUT_JSON = HERE / "mfa_vocals_whisper_segments.json"
    OUT_PHONE_JSON = HERE / "mfa_vocals_phone_segments.json"
    OUT_FLAT_PHONE_JSON = HERE / "mfa_vocals_phone_timestamps.json"

    def run(cmd, capture=False):
        return subprocess.run(cmd, check=True, text=True, capture_output=capture)


    # ---- deterministic clean slate ----
    if force_fresh_run:
        if CORPUS.exists():
            shutil.rmtree(CORPUS)
        if OUTDIR.exists():
            shutil.rmtree(OUTDIR)
    CORPUS.mkdir(exist_ok=True)
    OUTDIR.mkdir(exist_ok=True)

    # ---- normalization ----
    _word_clean_re = re.compile(r"[^a-z0-9']+")
    def norm(w: str) -> str:
        w = (w or "").strip().lower().replace("’", "'")
        w = unicodedata.normalize("NFKD", w)
        w = w.encode("ascii", "ignore").decode("ascii")
        w = _word_clean_re.sub("", w)
        return w

    def is_good_token(raw: str) -> bool:
        nw = norm(raw)
        if not nw:
            return False
        if nw.isdigit():
            return False
        # Drop 1-char junk but keep real ones used in English
        if len(nw) == 1 and nw not in {"a", "i", "o"}:
            return False
        return True

    def build_fuzzy_mapping(whisper_norm_list, mfa_norm_list, max_lookahead):
        mapping = [None] * len(whisper_norm_list)
        i = 0
        j = 0
        while i < len(whisper_norm_list) and j < len(mfa_norm_list):
            if whisper_norm_list[i] == mfa_norm_list[j]:
                mapping[i] = j
                i += 1
                j += 1
                continue

            found = None
            for di in range(0, max_lookahead + 1):
                for dj in range(0, max_lookahead + 1):
                    wi = i + di
                    mj = j + dj
                    if wi < len(whisper_norm_list) and mj < len(mfa_norm_list):
                        if whisper_norm_list[wi] == mfa_norm_list[mj]:
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

            if i + 1 < len(whisper_norm_list) and whisper_norm_list[i + 1] == mfa_norm_list[j]:
                mapping[i] = None
                i += 1
                continue
            if j + 1 < len(mfa_norm_list) and whisper_norm_list[i] == mfa_norm_list[j + 1]:
                j += 1
                continue

            mapping[i] = None
            i += 1
            j += 1
        return mapping

    # ---- 1) load whisper segments ----
    segments = json.loads(WHISPER_JSON.read_text(encoding="utf-8"))

    whisper_word_items = []
    for seg in segments:
        for w in seg.get("words", []):
            whisper_word_items.append({
                "word": w.get("word", ""),
                "start": w.get("start"),
                "end": w.get("end"),
            })

    whisper_words_raw = [item["word"] for item in whisper_word_items]

    # Build the exact token stream sent to MFA and keep a back-reference to original words.
    transcript_entries = []
    for idx, item in enumerate(whisper_word_items):
        raw_word = item["word"]
        if drop_weird_tokens and not is_good_token(raw_word):
            continue
        normalized = norm(raw_word)
        if not normalized:
            continue
        transcript_entries.append({
            "orig_index": idx,
            "norm": normalized,
            "start": item["start"],
            "end": item["end"],
        })

    if not transcript_entries:
        raise RuntimeError("No usable words found in whisper JSON after cleaning.")

    # ---- 2) convert to wav (recommended by MFA docs) ----
    run(["ffmpeg", "-y", "-i", str(AUDIO_MP3), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(WAV)])

    # ---- 3) Optional: OOV handling (replace unknown with oov_token if dict supports it) ----
    # We cannot reliably locate the model dictionary path when you pass a model name,
    # so we do a conservative approach:
    # - if `dictionary` looks like a file path, load vocab from it
    # - else, we can only do token cleanup (still helps a lot)
    vocab = None
    oov_ok = False
    dict_path = Path(dictionary)
    if dict_path.exists() and dict_path.is_file():
        vocab = set()
        with dict_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # MFA dict format: WORD  PHONEMES...
                vocab.add(line.split()[0].lower())
        oov_ok = (oov_token in vocab)

    if vocab is not None and oov_ok:
        for entry in transcript_entries:
            entry["token"] = entry["norm"] if entry["norm"] in vocab else oov_token
    else:
        for entry in transcript_entries:
            entry["token"] = entry["norm"]

    # ---- 4) Chunk into multiple utterances to prevent catastrophic drift ----
    # MFA aligns per file; easiest chunking is to create multiple (wav,txt) pairs
    # by slicing the wav using ffmpeg and writing per-chunk transcript.
    #
    # We chunk by character budget (works well for songs where timestamps can drift).
    # If you want time-based chunking, we can do that too.
    chunks = []
    cur = []
    cur_chars = 0
    for entry in transcript_entries:
        tok = entry["token"]
        add = len(tok) + (1 if cur else 0)
        if cur and (cur_chars + add) > max_chars_per_chunk:
            chunks.append(cur)
            cur = [entry]
            cur_chars = len(tok)
        else:
            cur.append(entry)
            cur_chars += add
    if cur:
        chunks.append(cur)

    def has_valid_time(entry):
        try:
            start = float(entry["start"])
            end = float(entry["end"])
            return end > start
        except (TypeError, ValueError):
            return False

    can_time_slice = all(has_valid_time(entry) for entry in transcript_entries)
    if not can_time_slice and len(chunks) > 1:
        print("Warning: missing/invalid Whisper word timestamps; falling back to single-chunk MFA alignment.")
        chunks = [transcript_entries]

    n_chunks = len(chunks)

    # Build corpus with numbered files
    # e.g. vocals_000.wav + vocals_000.txt ...
    corpus_pairs = []
    for i, chunk_entries in enumerate(chunks):
        stem = f"vocals_{i:03d}"
        wav_i = CORPUS / f"{stem}.wav"
        txt_i = CORPUS / f"{stem}.txt"
        txt_i.write_text(" ".join(entry["token"] for entry in chunk_entries) + "\n", encoding="utf-8")

        if n_chunks == 1:
            # reuse original wav (no slicing)
            if not wav_i.exists():
                # hardlink/copy is fine; copy avoids FS edge cases
                shutil.copy2(WAV, wav_i)
            chunk_offset = 0.0
        else:
            # Slice using the same word-time range this chunk text came from.
            start = float(chunk_entries[0]["start"])
            end = float(chunk_entries[-1]["end"])
            start2 = max(0.0, start - (chunk_silence_gap_s if i > 0 else 0.0))
            end2 = max(start2 + 0.01, end + (chunk_silence_gap_s if i < n_chunks - 1 else 0.0))
            run([
                "ffmpeg", "-y",
                "-i", str(WAV),
                "-ss", f"{start2:.3f}",
                "-to", f"{end2:.3f}",
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                str(wav_i)
            ])
            chunk_offset = start2

        corpus_pairs.append((wav_i, txt_i, chunk_offset))

    # ---- 5) Run MFA align ----
    # Accept either friendly aliases (e.g. "english") or explicit MFA model IDs.
    if acoustic in ACOUSTIC_MODELS:
        acoustic = ACOUSTIC_MODELS[acoustic]
    if dictionary in DICTIONARY_MODELS:
        dictionary = DICTIONARY_MODELS[dictionary]

    cmd = [
        "conda", "run", "-n", "mfa",
        "mfa", "align",
        str(CORPUS),
        dictionary,
        acoustic,
        str(OUTDIR),
        "--clean",
        "--overwrite",
    ]
    # determinism / stability
    if num_jobs is not None:
        cmd += ["--num_jobs", str(int(num_jobs))]

    def looks_like_missing_model(exc: subprocess.CalledProcessError) -> bool:
        stderr = (exc.stderr or "").lower()
        stdout = (exc.stdout or "").lower()
        combined = f"{stderr}\n{stdout}"
        markers = [
            "could not find",
            "not found",
            "does not exist",
            "missing model",
            "acoustic model",
            "dictionary model",
        ]
        return any(marker in combined for marker in markers)

    def tail(text: str, n: int = 1200) -> str:
        if not text:
            return ""
        return text[-n:]

    try:
        run(cmd, capture=True)
    except subprocess.CalledProcessError as exc:
        if not looks_like_missing_model(exc):
            raise RuntimeError(
                "MFA align failed (not a missing-model error).\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr (tail):\n{tail(exc.stderr)}"
            ) from exc

        install_lang(acoustic, dictionary)
        try:
            run(cmd, capture=True)
        except subprocess.CalledProcessError as exc2:
            raise RuntimeError(
                "MFA align failed after attempting model download.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr (tail):\n{tail(exc2.stderr)}"
            ) from exc2
        
        

    # ---- 6) Parse TextGrids and concatenate word/phone intervals (with offsets) ----
    # MFA may occasionally export only a subset of chunk TextGrids; skip missing chunks.
    all_intervals = []
    all_phone_intervals = []
    missing_chunks = []
    for i, (wav_i, _txt_i, chunk_offset) in enumerate(corpus_pairs):
        tg_path = OUTDIR / f"{wav_i.stem}.TextGrid"
        if not tg_path.exists():
            # Some MFA versions output in subdirs; try a glob fallback
            matches = list(OUTDIR.glob(f"**/{wav_i.stem}.TextGrid"))
            if matches:
                tg_path = matches[0]
            else:
                # Last resort: find any TextGrid with same numeric suffix (e.g. vocals_003)
                suffix = wav_i.stem.split("_")[-1]
                suffix_matches = list(OUTDIR.glob(f"**/*_{suffix}.TextGrid"))
                if suffix_matches:
                    tg_path = suffix_matches[0]
                else:
                    missing_chunks.append((i, wav_i.stem))
                    continue

        tg = tgio.openTextgrid(str(tg_path), False)
        tier_names = tg.tierNameList

        word_tier_name = next((n for n in tier_names if _looks_like_word_tier(n)), None)
        if not word_tier_name:
            raise RuntimeError(f"Could not find a word tier in {tg_path.name}. Tiers: {tier_names}")

        phone_tier_name = next((n for n in tier_names if _looks_like_phone_tier(n)), None)
        if not phone_tier_name:
            raise RuntimeError(f"Could not find a phone tier in {tg_path.name}. Tiers: {tier_names}")

        word_tier = tg.tierDict[word_tier_name]
        intervals = [(float(s), float(e), lab.strip()) for (s, e, lab) in word_tier.entryList]
        intervals = [(s, e, lab) for (s, e, lab) in intervals if lab and lab.lower() not in {"sp", "sil"}]

        phone_tier = tg.tierDict[phone_tier_name]
        phone_intervals = [(float(s), float(e), lab.strip()) for (s, e, lab) in phone_tier.entryList]
        phone_intervals = [
            (s, e, lab) for (s, e, lab) in phone_intervals
            if lab and lab.lower() not in {"sp", "sil"}
        ]

        # Offset by the actual slice start time used for this chunk.
        offset = chunk_offset

        for s, e, lab in intervals:
            all_intervals.append((s + offset, e + offset, lab))

        for s, e, lab in phone_intervals:
            all_phone_intervals.append((s + offset, e + offset, lab))

    if missing_chunks:
        missing_txt = ", ".join(f"{stem}(#{idx})" for idx, stem in missing_chunks)
        print(f"Warning: skipped missing TextGrids: {missing_txt}")

    if not all_intervals:
        raise RuntimeError("No MFA word intervals were parsed from TextGrid output.")

    # ---- 7) map original whisper words to MFA words ----
    expected_norm = [entry["token"] for entry in transcript_entries]
    mfa_norm = [norm(lab) for _, _, lab in all_intervals]

    if expected_norm != mfa_norm:
        if not allow_fuzzy:
            n = min(len(expected_norm), len(mfa_norm))
            first_bad = next((i for i in range(n) if expected_norm[i] != mfa_norm[i]), None)
            raise RuntimeError(
                "MFA word sequence != Whisper word sequence.\n"
                f"Whisper words sent to MFA: {len(expected_norm)} | MFA words: {len(mfa_norm)}\n"
                f"First mismatch index: {first_bad}\n"
                "Fix: re-run with allow_fuzzy=True (recommended for songs)."
            )
        filtered_mapping = build_fuzzy_mapping(expected_norm, mfa_norm, fuzzy_max_lookahead)
    else:
        filtered_mapping = list(range(len(expected_norm)))

    if all(m is None for m in filtered_mapping):
        raise RuntimeError(
            "Fuzzy alignment failed to match any words. "
            "Try smaller max_chars_per_chunk, or provide a dictionary path so OOV replacement can work."
        )

    # Expand filtered-token mapping back to full Whisper word index space.
    mapping = [None] * len(whisper_words_raw)
    for i, entry in enumerate(transcript_entries):
        mapping[entry["orig_index"]] = filtered_mapping[i]

    # Build phone groups aligned to each MFA word interval so UI can highlight sub-word parts.
    phones_by_mfa_word_index = {}
    for wi, (ws, we, _wlab) in enumerate(all_intervals):
        grouped = []
        for ps, pe, plab in all_phone_intervals:
            if pe <= ws or ps >= we:
                continue
            grouped.append({
                "phone": plab,
                "start": max(ps, ws),
                "end": min(pe, we),
            })
        phones_by_mfa_word_index[wi] = grouped

    # ---- 8) overwrite whisper timings with MFA timings ----
    k = 0
    for seg in segments:
        ws = seg.get("words", [])
        if not ws:
            continue
        for w in ws:
            mfa_index = mapping[k] if k < len(mapping) else None
            if mfa_index is not None and mfa_index < len(all_intervals):
                s, e, _lab = all_intervals[mfa_index]
                w["start"] = s
                w["end"] = e
            k += 1
        seg["start"] = ws[0].get("start", seg.get("start"))
        seg["end"] = ws[-1].get("end", seg.get("end"))

    OUT_JSON.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    phone_segments = copy.deepcopy(segments)
    k = 0
    for seg in phone_segments:
        ws = seg.get("words", [])
        if not ws:
            continue
        for w in ws:
            mfa_index = mapping[k] if k < len(mapping) else None
            if mfa_index is not None and mfa_index in phones_by_mfa_word_index:
                w["phones"] = phones_by_mfa_word_index.get(mfa_index, [])
            else:
                w["phones"] = []
            k += 1
        seg["start"] = ws[0].get("start", seg.get("start"))
        seg["end"] = ws[-1].get("end", seg.get("end"))

    effective_phonemizer_language = phonemizer_language or _infer_phonemizer_language(dictionary, acoustic)
    enrich_stats = _enrich_words_with_phone_segments(
        phone_segments,
        phonemizer_language=effective_phonemizer_language,
        use_phonemizer=bool(use_phonemizer),
    )

    OUT_PHONE_JSON.write_text(json.dumps(phone_segments, ensure_ascii=False, indent=2), encoding="utf-8")

    if export_flat_phone_json:
        flat_phone_payload = [
            {"phone": lab, "start": s, "end": e}
            for (s, e, lab) in all_phone_intervals
        ]
        OUT_FLAT_PHONE_JSON.write_text(
            json.dumps(flat_phone_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Wrote", OUT_JSON)
    print("Wrote", OUT_PHONE_JSON)
    print(
        "Phone enrichment:",
        f"fallback_words={enrich_stats['words_filled_fallback']}/{enrich_stats['words_total']}",
        f"phonemizer={enrich_stats['phonemizer_enabled']}",
        f"lang={enrich_stats['phonemizer_language']}",
    )
    if export_flat_phone_json:
        print("Wrote", OUT_FLAT_PHONE_JSON)


def phone_word_coverage(phone_json_path) -> float:
    p = Path(phone_json_path)
    if not p.exists():
        return 0.0

    data = json.loads(p.read_text(encoding="utf-8"))
    total = 0
    with_phones = 0
    for seg in data:
        for w in seg.get("words", []):
            total += 1
            if w.get("phones"):
                with_phones += 1
    if total == 0:
        return 0.0
    return (with_phones / total) * 100.0


def fill_missing_phones_from_char_alignments(
    head_folder,
    aligned_chars_json="vocals_whisper_segments_wav2vec2.json",
    phone_json="mfa_vocals_phone_segments.json",
    base_json="mfa_vocals_whisper_segments.json",
    out_json="mfa_vocals_phone_segments.json",
    phonemizer_language: Optional[str] = None,
    use_phonemizer=True,
):
    """
    Fill missing `word.phones` entries using WhisperX character alignments.
    This is a wav2vec2-family fallback path (WhisperX CTC align model).
    """

    here = Path(head_folder)
    aligned_path = here / aligned_chars_json
    phone_path = here / phone_json
    base_path = here / base_json
    out_path = here / out_json

    if not aligned_path.exists():
        raise RuntimeError(f"Missing char alignment file: {aligned_path}")

    if phone_path.exists():
        target = json.loads(phone_path.read_text(encoding="utf-8"))
    elif base_path.exists():
        target = json.loads(base_path.read_text(encoding="utf-8"))
    else:
        raise RuntimeError(f"Missing source transcript for phone fallback: {phone_path} / {base_path}")

    aligned_data = json.loads(aligned_path.read_text(encoding="utf-8"))
    aligned_segments = aligned_data.get("segments", []) if isinstance(aligned_data, dict) else aligned_data

    def _chars_for_segment(seg_obj):
        chars = []
        for c in seg_obj.get("chars", []) or []:
            ch = str(c.get("char", "")).strip()
            if not ch:
                continue
            if "start" not in c or "end" not in c:
                continue
            chars.append({
                "phone": ch,
                "start": float(c["start"]),
                "end": float(c["end"]),
            })
        return chars

    aligned_by_id = {}
    for i, seg in enumerate(aligned_segments):
        seg_id = seg.get("id", i)
        aligned_by_id[seg_id] = seg

    filled_words = 0
    missing_words = 0

    for i, seg in enumerate(target):
        seg_id = seg.get("id", i)
        aligned_seg = aligned_by_id.get(seg_id, aligned_segments[i] if i < len(aligned_segments) else {})
        seg_chars = _chars_for_segment(aligned_seg)

        for w in seg.get("words", []):
            if w.get("phones"):
                continue
            missing_words += 1

            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            overlap = []
            for c in seg_chars:
                if c["end"] <= ws or c["start"] >= we:
                    continue
                overlap.append({
                    "phone": c["phone"],
                    "start": max(ws, c["start"]),
                    "end": min(we, c["end"]),
                })

            if not overlap:
                overlap = []

            if overlap:
                w["phones"] = overlap
                filled_words += 1

    enrich_stats = _enrich_words_with_phone_segments(
        target,
        phonemizer_language=phonemizer_language or "en-us",
        use_phonemizer=bool(use_phonemizer),
    )

    out_path.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "filled_words": filled_words,
        "missing_words_before": missing_words,
        "coverage_after": phone_word_coverage(out_path),
        "output": str(out_path),
        "fallback_words": enrich_stats["words_filled_fallback"],
        "phonemizer_enabled": enrich_stats["phonemizer_enabled"],
        "phonemizer_language": enrich_stats["phonemizer_language"],
    }


def enrich_phone_json_with_text_segments(
    phone_json_path,
    out_json: Optional[str] = None,
    phonemizer_language: Optional[str] = None,
    use_phonemizer=True,
):
    src = Path(phone_json_path)
    if not src.exists():
        raise RuntimeError(f"Missing phone JSON: {src}")

    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Expected top-level list of segments in phone JSON")

    stats = _enrich_words_with_phone_segments(
        data,
        phonemizer_language=phonemizer_language or "en-us",
        use_phonemizer=bool(use_phonemizer),
    )

    out_path = Path(out_json) if out_json else src
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    print(
        "Phone enrichment:",
        f"fallback_words={stats['words_filled_fallback']}/{stats['words_total']}",
        f"phonemizer={stats['phonemizer_enabled']}",
        f"lang={stats['phonemizer_language']}",
    )
    return stats


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Run MFA alignment and export phone-level timestamps.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    align_cmd = sub.add_parser(
        "align",
        help="Run MFA alignment and emit word + phone timestamp JSON files.",
    )
    align_cmd.add_argument("head_folder", help="Folder with vocals.mp3 and vocals_whisper_segments.json")
    align_cmd.add_argument("--vocals", default="vocals.mp3")
    align_cmd.add_argument("--transcript", default="vocals_whisper_segments.json")
    align_cmd.add_argument("--acoustic", default="english_us_arpa")
    align_cmd.add_argument("--dictionary", default="english_us_arpa")
    align_cmd.add_argument("--allow-fuzzy", action="store_true")
    align_cmd.add_argument("--fuzzy-max-lookahead", type=int, default=8)
    align_cmd.add_argument("--max-chars-per-chunk", type=int, default=220)
    align_cmd.add_argument("--num-jobs", type=int, default=1)
    align_cmd.add_argument("--no-flat-phone-json", action="store_true")
    align_cmd.add_argument("--phonemizer-lang", default=None)
    align_cmd.add_argument("--no-phonemizer", action="store_true")

    export_cmd = sub.add_parser(
        "export-phones",
        help="Export flat phone timestamps from an existing MFA TextGrid.",
    )
    export_cmd.add_argument("textgrid", help="Path to MFA TextGrid file")
    export_cmd.add_argument("--out", default=None, help="Output JSON path (defaults next to TextGrid)")
    export_cmd.add_argument("--include-silence", action="store_true")

    segment_cmd = sub.add_parser(
        "segment-phones",
        help="Ensure every word has phones and add per-word phone_segments in an existing JSON.",
    )
    segment_cmd.add_argument("phone_json", help="Path to mfa_vocals_phone_segments.json (or compatible file)")
    segment_cmd.add_argument("--out", default=None, help="Output JSON path (default: overwrite input)")
    segment_cmd.add_argument("--phonemizer-lang", default=None)
    segment_cmd.add_argument("--no-phonemizer", action="store_true")

    return parser


def _main():
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command == "align":
        generate_aligned_v2(
            args.head_folder,
            vocals=args.vocals,
            transcript=args.transcript,
            acoustic=args.acoustic,
            dictionary=args.dictionary,
            allow_fuzzy=bool(args.allow_fuzzy),
            fuzzy_max_lookahead=int(args.fuzzy_max_lookahead),
            max_chars_per_chunk=int(args.max_chars_per_chunk),
            num_jobs=int(args.num_jobs),
            export_flat_phone_json=not bool(args.no_flat_phone_json),
            phonemizer_language=args.phonemizer_lang,
            use_phonemizer=not bool(args.no_phonemizer),
        )
        return

    if args.command == "export-phones":
        export_phone_timestamps_from_textgrid(
            args.textgrid,
            out_json=args.out,
            include_silence=bool(args.include_silence),
        )
        return

    if args.command == "segment-phones":
        enrich_phone_json_with_text_segments(
            args.phone_json,
            out_json=args.out,
            phonemizer_language=args.phonemizer_lang,
            use_phonemizer=not bool(args.no_phonemizer),
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    _main()