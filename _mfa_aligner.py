import json, re, subprocess
import difflib
import unicodedata
from pathlib import Path
from praatio import tgio
from typing import Optional

#HERE = Path(__file__).resolve().parent
def generate_aligned(head_folder, vocals="vocals.mp3", transcript="vocals_whisper_segments.json", acoustic="english_us_arpa", dictionary="english_us_arpa", allow_fuzzy=False):
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

    def build_fuzzy_mapping(whisper_norm_list, mfa_norm_list):
        matcher = difflib.SequenceMatcher(a=whisper_norm_list, b=mfa_norm_list)
        mapping = [None] * len(whisper_norm_list)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    mapping[i] = j
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
        mapping = build_fuzzy_mapping(whisper_norm, mfa_norm)
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
    def run(cmd):
        subprocess.run(cmd, check=True)
    run(["mfa", "model", "download", "acoustic", model_general])
    if dictionary:
        run(["mfa", "model", "download", "dictionary", dictionary])
    else:
        run(["mfa", "model", "download", "dictionary", model_general])