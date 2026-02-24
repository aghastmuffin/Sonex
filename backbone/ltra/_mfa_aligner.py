import json, re, subprocess
import unicodedata
from pathlib import Path
from praatio import tgio
from typing import Optional

import json, re, subprocess, shutil, tempfile
import unicodedata
from pathlib import Path
from praatio import tgio
from typing import Optional

#HERE = Path(__file__).resolve().parent
def generate_aligned(head_folder, vocals="vocals.mp3", transcript="vocals_whisper_segments.json", acoustic="english_us_arpa", dictionary="english_us_arpa", allow_fuzzy=False, fuzzy_max_lookahead=6):
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
    def run(cmd):
        subprocess.run(cmd, check=True)
    run(["mfa", "model", "download", "acoustic", model_general])
    if dictionary:
        run(["mfa", "model", "download", "dictionary", dictionary])
    else:
        run(["mfa", "model", "download", "dictionary", model_general])

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
):
    """
    More robust MFA aligner:
    - normalizes + filters Whisper tokens BEFORE transcript creation
    - optional OOV handling: replaces unknown words w/ `oov_token` (if present in dict)
    - chunks long transcripts into multiple utterances to prevent catastrophic drift
    - forces fresh runs by deleting corpus/outdir to remove nondeterministic “random fixes”
    - optionally runs with num_jobs=1 for determinism

    Output file is the same as v1: mfa_vocals_whisper_segments.json
    """

    HERE = Path(head_folder)
    AUDIO_MP3 = HERE / vocals
    WHISPER_JSON = HERE / transcript

    CORPUS = HERE / "_mfa_corpus"
    OUTDIR = HERE / "_mfa_out"
    WAV = CORPUS / "vocals.wav"
    OUT_JSON = HERE / "mfa_vocals_whisper_segments.json"

    def run(cmd):
        subprocess.run(cmd, check=True)

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

    whisper_words_raw = []
    for seg in segments:
        for w in seg.get("words", []):
            whisper_words_raw.append(w.get("word", ""))

    if drop_weird_tokens:
        whisper_words_raw = [w for w in whisper_words_raw if is_good_token(w)]
    whisper_norm_words = [norm(w) for w in whisper_words_raw if norm(w)]
    if not whisper_norm_words:
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

    transcript_tokens = []
    if vocab is not None and oov_ok:
        for w in whisper_norm_words:
            transcript_tokens.append(w if w in vocab else oov_token)
    else:
        # fall back to cleaned normalized tokens (still big improvement vs raw whisper tokens)
        transcript_tokens = whisper_norm_words

    # ---- 4) Chunk into multiple utterances to prevent catastrophic drift ----
    # MFA aligns per file; easiest chunking is to create multiple (wav,txt) pairs
    # by slicing the wav using ffmpeg and writing per-chunk transcript.
    #
    # We chunk by character budget (works well for songs where timestamps can drift).
    # If you want time-based chunking, we can do that too.
    chunks = []
    cur = []
    cur_chars = 0
    for tok in transcript_tokens:
        add = len(tok) + (1 if cur else 0)
        if cur and (cur_chars + add) > max_chars_per_chunk:
            chunks.append(cur)
            cur = [tok]
            cur_chars = len(tok)
        else:
            cur.append(tok)
            cur_chars += add
    if cur:
        chunks.append(cur)

    # If it's just one chunk, behave like v1 but with cleaned transcript
    # Otherwise, create multi-utterance corpus and align each chunk audio slice.
    # We slice audio evenly by duration; this is not perfect but prevents total failure.
    # If you want, we can instead slice using Whisper segment times (even better).
    # (No language correction is done here; you control models externally.)
    #
    # Get wav duration via ffprobe
    def wav_duration_seconds(path: Path) -> float:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ])
        return float(out.decode("utf-8").strip())

    total_dur = wav_duration_seconds(WAV)
    n_chunks = len(chunks)

    # Build corpus with numbered files
    # e.g. vocals_000.wav + vocals_000.txt ...
    corpus_pairs = []
    for i, toks in enumerate(chunks):
        stem = f"vocals_{i:03d}"
        wav_i = CORPUS / f"{stem}.wav"
        txt_i = CORPUS / f"{stem}.txt"
        txt_i.write_text(" ".join(toks) + "\n", encoding="utf-8")

        if n_chunks == 1:
            # reuse original wav (no slicing)
            if not wav_i.exists():
                # hardlink/copy is fine; copy avoids FS edge cases
                shutil.copy2(WAV, wav_i)
        else:
            # slice evenly across duration
            start = (total_dur * i) / n_chunks
            end = (total_dur * (i + 1)) / n_chunks
            # add a tiny gap trimming to reduce boundary bleed
            start2 = max(0.0, start + (chunk_silence_gap_s if i > 0 else 0.0))
            end2 = max(start2, end - (chunk_silence_gap_s if i < n_chunks - 1 else 0.0))
            run([
                "ffmpeg", "-y",
                "-i", str(WAV),
                "-ss", f"{start2:.3f}",
                "-to", f"{end2:.3f}",
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                str(wav_i)
            ])

        corpus_pairs.append((wav_i, txt_i))

    # ---- 5) Run MFA align ----
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

    run(cmd)

    # ---- 6) Parse all TextGrids and concatenate word intervals (with offsets) ----
    # MFA outputs one TextGrid per wav in OUTDIR, usually matching filenames.
    all_intervals = []
    offset = 0.0
    for i, (wav_i, _txt_i) in enumerate(corpus_pairs):
        tg_path = OUTDIR / f"{wav_i.stem}.TextGrid"
        if not tg_path.exists():
            # Some MFA versions output in subdirs; try a glob fallback
            matches = list(OUTDIR.glob(f"**/{wav_i.stem}.TextGrid"))
            if matches:
                tg_path = matches[0]
            else:
                raise RuntimeError(f"Missing TextGrid for chunk {i}: expected {wav_i.stem}.TextGrid")

        tg = tgio.openTextgrid(str(tg_path), False)
        tier_names = tg.tierNameList

        def looks_like_word_tier(name: str) -> bool:
            n = name.strip().lower()
            return n == "word" or n == "words" or n.endswith(" - words") or "words" in n or "word" in n

        word_tier_name = next((n for n in tier_names if looks_like_word_tier(n)), None)
        if not word_tier_name:
            raise RuntimeError(f"Could not find a word tier in {tg_path.name}. Tiers: {tier_names}")

        word_tier = tg.tierDict[word_tier_name]
        intervals = [(float(s), float(e), lab.strip()) for (s, e, lab) in word_tier.entryList]
        intervals = [(s, e, lab) for (s, e, lab) in intervals if lab and lab.lower() not in {"sp", "sil"}]

        # Offset by chunk start time (only meaningful when we sliced)
        if n_chunks > 1:
            start = (total_dur * i) / n_chunks
            start2 = max(0.0, start + (chunk_silence_gap_s if i > 0 else 0.0))
            offset = start2
        else:
            offset = 0.0

        for s, e, lab in intervals:
            all_intervals.append((s + offset, e + offset, lab))

    # ---- 7) map original whisper words to MFA words ----
    whisper_norm = [norm(w) for w in whisper_words_raw]
    mfa_norm = [norm(lab) for _, _, lab in all_intervals]

    # If we replaced OOVs with oov_token, we should apply the same replacement in whisper_norm
    if vocab is not None and oov_ok:
        whisper_norm = [(w if w in vocab else oov_token) for w in whisper_norm]

    if whisper_norm != mfa_norm:
        if not allow_fuzzy:
            n = min(len(whisper_norm), len(mfa_norm))
            first_bad = next((i for i in range(n) if whisper_norm[i] != mfa_norm[i]), None)
            raise RuntimeError(
                "MFA word sequence != Whisper word sequence.\n"
                f"Whisper words: {len(whisper_norm)} | MFA words: {len(mfa_norm)}\n"
                f"First mismatch index: {first_bad}\n"
                "Fix: re-run with allow_fuzzy=True (recommended for songs)."
            )
        mapping = build_fuzzy_mapping(whisper_norm, mfa_norm, fuzzy_max_lookahead)
        if all(m is None for m in mapping):
            raise RuntimeError(
                "Fuzzy alignment failed to match any words. "
                "Try smaller max_chars_per_chunk, or provide a dictionary path so OOV replacement can work."
            )
    else:
        mapping = list(range(len(whisper_norm)))

    # ---- 8) overwrite whisper timings with MFA timings ----
    k = 0
    for seg in segments:
        ws = seg.get("words", [])
        if not ws:
            continue
        for w in ws:
            mfa_index = mapping[k]
            if mfa_index is not None:
                s, e, _lab = all_intervals[mfa_index]
                w["start"] = s
                w["end"] = e
            k += 1
        seg["start"] = ws[0].get("start", seg.get("start"))
        seg["end"] = ws[-1].get("end", seg.get("end"))

    OUT_JSON.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)