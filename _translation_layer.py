# _translation_layer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------
# Model caches (avoid reload)
# ----------------------------
_model_cache: Dict[str, Any] = {}
_tokenizer_cache: Dict[str, Any] = {}

# ---- Global singleton cache ----
_NLLB_TOKENIZER = None
_NLLB_MODEL = None
_NLLB_DEVICE = None

# ----------------------------
# Public API (drop-in)
# ----------------------------

def _get_device() -> torch.device:
    # Works for Mac/Windows/Linux; uses CUDA if available, otherwise CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_nllb(
    model_name: str = "facebook/nllb-200-distilled-600M",
):
    global _NLLB_TOKENIZER, _NLLB_MODEL, _NLLB_DEVICE

    if _NLLB_MODEL is not None and _NLLB_TOKENIZER is not None:
        return _NLLB_TOKENIZER, _NLLB_MODEL, _NLLB_DEVICE

    device = _get_device()

    tok = AutoTokenizer.from_pretrained(model_name)
    # fp16 on CUDA speeds things up without meaningfully hurting quality for MT
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    _NLLB_TOKENIZER = tok
    _NLLB_MODEL = model
    _NLLB_DEVICE = device
    return tok, model, device


def nllbtranslate(
    text: str,
    target_to: str,
    target_from: str,
    *,
    max_new_tokens: int = 256,
    num_beams: int = 4,          # keep quality (same as typical “good” default)
    batch_size: int = 8,         # unused here, but kept for signature compatibility if you want
) -> str:
    """
    Single-string translate. Still uses global-cached model, so it's fast to call repeatedly.
    """
    text = (text or "").strip()
    if not text:
        return ""

    tok, model, device = _load_nllb()

    tok.src_lang = target_from
    inputs = tok([text], return_tensors="pt", padding=True, truncation=True).to(device)

    forced_id = tok.convert_tokens_to_ids(target_to)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            forced_bos_token_id=forced_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    return _normalize_spaces(tok.batch_decode(out, skip_special_tokens=True)[0])


def nllbtranslate_many(
    texts: List[str],
    target_to: str,
    target_from: str,
    *,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    num_beams: int = 4,          # keep quality
) -> List[str]:
    """
    Batched translate: MASSIVE speedup vs calling nllbtranslate() in a loop.
    """
    if not texts:
        return []

    tok, model, device = _load_nllb()
    tok.src_lang = target_from
    forced_id = tok.convert_tokens_to_ids(target_to)

    results: List[str] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        inputs = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                forced_bos_token_id=forced_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

        decoded = tok.batch_decode(out, skip_special_tokens=True)
        results.extend([_normalize_spaces(s) for s in decoded])

    return results


def translate_segment_words_time_synced(
    source_words: List[Dict[str, Any]],
    translated_text: str,
    seg_start: float,
    seg_end: float,
    merge_contractions: bool = True,
) -> List[Dict[str, Any]]:
    """
    Core fix:
    Given *source* word timestamps, produce *target* word timestamps that:
      - stay within the segment [seg_start, seg_end]
      - preserve order
      - if a single source word translates to multiple target words,
        the source word's time is split across those target words.

    Output format matches Whisper-style:
      [{"word": "...", "start": 0.0, "end": 0.1}, ...]
    """
    # If no source word timings exist, fall back to uniform split across segment.
    src = _sanitize_words(source_words)
    tgt_tokens = _tokenize_for_timing(translated_text, merge_contractions=merge_contractions)

    if not tgt_tokens:
        return []

    if not src:
        return _uniform_time_tokens(tgt_tokens, seg_start, seg_end)

    # Assign target tokens across source words.
    # We do this in 2 stages:
    #  1) Decide how many target tokens belong to each source word (counts per src index)
    #  2) For each source word, split its [start,end] across its assigned tokens
    counts = _allocate_tokens_to_source_words(src, tgt_tokens)

    out: List[Dict[str, Any]] = []
    ti = 0
    for si, sw in enumerate(src):
        n = counts[si]
        if n <= 0:
            continue
        chunk = tgt_tokens[ti : ti + n]
        ti += n

        sw_start = float(sw["start"])
        sw_end = float(sw["end"])
        # Safety clamp
        sw_start = max(sw_start, float(seg_start))
        sw_end = min(sw_end, float(seg_end))
        if sw_end <= sw_start:
            sw_end = min(float(seg_end), sw_start + 1e-3)

        out.extend(_split_interval_across_tokens(chunk, sw_start, sw_end))

    # If anything left (rounding/edge cases), tack it onto the end.
    if ti < len(tgt_tokens):
        last_end = out[-1]["end"] if out else float(seg_start)
        extra = tgt_tokens[ti:]
        out.extend(_split_interval_across_tokens(extra, last_end, float(seg_end)))

    # Final clamp + monotonic fix
    out = _clamp_and_monotonic(out, float(seg_start), float(seg_end))
    return out


def retime_translation_segments(
    segments: List[Dict[str, Any]],
    target_words_key: str = "words",
    source_words_key: str = "source_words",
    merge_contractions: bool = True,
) -> List[Dict[str, Any]]:
    """
    Drop-in helper for your pipeline:
    Each segment must have:
      - "start", "end"
      - translated text in "text"
      - source words in `source_words_key` (Whisper words with timings)

    This replaces segment[target_words_key] with a re-timed version that
    uses the source word boundaries.
    """
    out = []
    for seg in segments:
        seg2 = dict(seg)
        seg_start = float(seg2.get("start", 0.0))
        seg_end = float(seg2.get("end", seg_start))
        translated_text = str(seg2.get("text", "") or "")

        src_words = seg2.get(source_words_key, []) or []
        seg2[target_words_key] = translate_segment_words_time_synced(
            src_words,
            translated_text,
            seg_start,
            seg_end,
            merge_contractions=merge_contractions,
        )
        out.append(seg2)
    return out


def segment_text_to_words(text: str, start: float, end: float) -> List[Dict[str, Any]]:
    """
    Backward-compat shim:
    If you previously created words by splitting translated text uniformly,
    this keeps the same behavior. (Your master script can call this, but
    the *real* fix is retime_translation_segments().)
    """
    toks = _tokenize_for_timing(text, merge_contractions=True)
    return _uniform_time_tokens(toks, float(start), float(end))


# ----------------------------
# Internals
# ----------------------------

def _normalize_spaces(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _sanitize_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for w in words or []:
        if "start" not in w or "end" not in w:
            continue
        try:
            ws = float(w["start"])
            we = float(w["end"])
        except Exception:
            continue
        if we <= ws:
            continue
        out.append({"word": str(w.get("word", "")).strip(), "start": ws, "end": we})
    return out

def _tokenize_for_timing(text: str, merge_contractions: bool = True) -> List[str]:
    """
    Tokenize into timing tokens:
      - words and punctuation become tokens
      - punctuation is kept, but later we’ll attach its timing to neighbors naturally
    """
    text = _normalize_spaces(text)
    if not text:
        return []

    # Split into words/punct as separate tokens
    # Example: "I'm here, love." -> ["I", "'m", "here", ",", "love", "."]
    tokens = re.findall(r"[A-Za-z0-9]+|['’][A-Za-z]+|[^\sA-Za-z0-9]", text)

    if merge_contractions:
        tokens = _merge_english_contractions(tokens)

    # Attach “lonely” punctuation to previous token as text (but keep it a token)
    # We'll keep punctuation tokens; timing allocation uses weights that ignore pure punct.
    return tokens

def _merge_english_contractions(tokens: List[str]) -> List[str]:
    """
    Merge patterns like ["I", "'m"] -> ["I'm"] and ["don", "'", "t"] -> ["don't"]
    Also handles curly apostrophe.
    """
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if i + 1 < len(tokens) and re.fullmatch(r"[A-Za-z]+", t) and re.fullmatch(r"['’][A-Za-z]+", tokens[i + 1]):
            out.append(t + tokens[i + 1])
            i += 2
            continue

        # don ' t  -> don't  (common from some tokenizers)
        if i + 2 < len(tokens) and re.fullmatch(r"[A-Za-z]+", t) and tokens[i + 1] in ["'", "’"] and re.fullmatch(r"[A-Za-z]+", tokens[i + 2]):
            out.append(t + tokens[i + 1] + tokens[i + 2])
            i += 3
            continue

        out.append(t)
        i += 1
    return out

def _uniform_time_tokens(tokens: List[str], start: float, end: float) -> List[Dict[str, Any]]:
    start = float(start)
    end = float(end)
    if end <= start:
        end = start + 1e-3
    n = max(1, len(tokens))
    dur = end - start
    step = dur / n
    out = []
    for i, tok in enumerate(tokens):
        s = start + i * step
        e = start + (i + 1) * step
        out.append({"word": tok, "start": s, "end": e})
    return out

def _token_weight(tok: str) -> float:
    # Weight ignores pure punctuation; prefers longer “wordy” tokens.
    if re.fullmatch(r"[^\w]+", tok):
        return 0.15
    # Remove punctuation for length measure
    core = re.sub(r"[^\w]+", "", tok)
    return max(0.4, float(len(core)))

def _allocate_tokens_to_source_words(
    src_words: List[Dict[str, Any]],
    tgt_tokens: List[str],
) -> List[int]:
    """
    Decide how many target tokens go to each source word.
    Goal: preserve timing faithfulness:
      - If tgt has more tokens than src, distribute extras across src words.
      - If tgt has fewer tokens than src, some src words get 0 (they'll be skipped),
        but overall the segment stays aligned.
    """
    S = len(src_words)
    T = len(tgt_tokens)
    if S <= 0:
        return []

    # Base: at least 1 token for as many source words as possible (if enough tokens)
    counts = [0] * S
    if T >= S:
        for i in range(S):
            counts[i] = 1
        remaining = T - S

        # Distribute remaining based on source word duration (longer source word -> more room)
        durs = [max(1e-3, float(w["end"]) - float(w["start"])) for w in src_words]
        total = sum(durs) or 1.0

        # First proportional allocation
        add = [int(round(remaining * (d / total))) for d in durs]
        # Fix rounding to exact remaining
        diff = remaining - sum(add)
        if diff != 0:
            # Adjust by descending duration
            order = sorted(range(S), key=lambda i: durs[i], reverse=True)
            step = 1 if diff > 0 else -1
            diff = abs(diff)
            j = 0
            while diff > 0 and order:
                idx = order[j % len(order)]
                if step < 0 and add[idx] <= 0:
                    j += 1
                    continue
                add[idx] += step
                diff -= 1
                j += 1

        for i in range(S):
            counts[i] += max(0, add[i])

        # Ensure sum matches T exactly (last resort)
        cur = sum(counts)
        if cur < T:
            counts[-1] += (T - cur)
        elif cur > T:
            # remove from end backwards
            extra = cur - T
            for i in range(S - 1, -1, -1):
                take = min(extra, max(0, counts[i] - 1))
                counts[i] -= take
                extra -= take
                if extra <= 0:
                    break
            # if still extra, force-remove (can hit 0)
            i = S - 1
            while extra > 0 and i >= 0:
                take = min(extra, counts[i])
                counts[i] -= take
                extra -= take
                i -= 1

        return counts

    # T < S: fewer target tokens than source words
    # Allocate 1 token to the longest-duration source words first.
    durs = [max(1e-3, float(w["end"]) - float(w["start"])) for w in src_words]
    order = sorted(range(S), key=lambda i: durs[i], reverse=True)
    for i in range(T):
        counts[order[i]] = 1

    # Keep order stability: move tokens left-to-right so earlier words get earlier tokens
    # Convert chosen indices to sorted and assign 1s in that order
    chosen = sorted([idx for idx, c in enumerate(counts) if c > 0])
    counts = [0] * S
    for idx in chosen:
        counts[idx] = 1
    return counts

def _split_interval_across_tokens(tokens: List[str], start: float, end: float) -> List[Dict[str, Any]]:
    start = float(start)
    end = float(end)
    if end <= start:
        end = start + 1e-3

    weights = [_token_weight(t) for t in tokens]
    total = sum(weights) or float(len(tokens))
    dur = end - start

    out = []
    t = start
    for i, tok in enumerate(tokens):
        w = weights[i]
        frac = (w / total) if total > 0 else (1.0 / len(tokens))
        dt = dur * frac
        s = t
        e = (end if i == len(tokens) - 1 else (t + dt))
        out.append({"word": tok, "start": s, "end": e})
        t = e
    return out

def _clamp_and_monotonic(words: List[Dict[str, Any]], seg_start: float, seg_end: float) -> List[Dict[str, Any]]:
    if not words:
        return words
    seg_start = float(seg_start)
    seg_end = float(seg_end)
    out = []
    prev_end = seg_start
    for w in words:
        s = max(seg_start, float(w["start"]))
        e = min(seg_end, float(w["end"]))
        if s < prev_end:
            s = prev_end
        if e <= s:
            e = min(seg_end, s + 1e-3)
        out.append({"word": str(w["word"]), "start": s, "end": e})
        prev_end = e
        if prev_end >= seg_end:
            break
    return out