# _translation_layer.py
from __future__ import annotations

from typing import Any, Dict, List
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------
# Global singleton cache
# ----------------------------
_NLLB_TOKENIZER = None
_NLLB_MODEL = None
_NLLB_DEVICE = None

# Optional speed knobs (safe)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------------------------
# Device + model loading
# ----------------------------
def _get_device() -> torch.device:
    """CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    global _NLLB_TOKENIZER, _NLLB_MODEL, _NLLB_DEVICE

    if _NLLB_MODEL is not None and _NLLB_TOKENIZER is not None and _NLLB_DEVICE is not None:
        return _NLLB_TOKENIZER, _NLLB_MODEL, _NLLB_DEVICE

    device = _get_device()

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # fp16 helps on CUDA; keep fp32 on MPS/CPU
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()

    _NLLB_TOKENIZER = tok
    _NLLB_MODEL = model
    _NLLB_DEVICE = device
    return tok, model, device


# ----------------------------
# Translation API
# ----------------------------
def _resolve_lang_token_id(tok: Any, lang_code: str) -> int:
    mapping = getattr(tok, "lang_code_to_id", None)
    if isinstance(mapping, dict) and lang_code in mapping:
        return int(mapping[lang_code])

    token_map = getattr(tok, "lang_code_to_token", None)
    if isinstance(token_map, dict) and lang_code in token_map:
        tok_id = tok.convert_tokens_to_ids(token_map[lang_code])
        if tok_id is not None and (tok.unk_token_id is None or tok_id != tok.unk_token_id):
            return int(tok_id)

    tok_id = tok.convert_tokens_to_ids(lang_code)
    if tok_id is not None and (tok.unk_token_id is None or tok_id != tok.unk_token_id):
        return int(tok_id)

    raise KeyError(f"Cannot resolve language code '{lang_code}' for tokenizer {type(tok).__name__}")


def nllbtranslate(
    text: str,
    target_to: str,
    target_from: str,
    *,
    max_new_tokens: int = 256,
    num_beams: int = 4,
) -> str:
    """
    Single-string translate. Uses global cached model/tokenizer.
    """
    text = (text or "").strip()
    if not text:
        return ""

    tok, model, device = _load_nllb()

    tok.src_lang = target_from

    forced_id = _resolve_lang_token_id(tok, target_to)

    inputs = tok([text], return_tensors="pt", padding=True, truncation=True).to(device)

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
    num_beams: int = 4,
) -> List[str]:
    """
    Batched translate. HUGE speedup vs calling nllbtranslate() in a loop.
    """
    if not texts:
        return []

    tok, model, device = _load_nllb()
    tok.src_lang = target_from
    forced_id = _resolve_lang_token_id(tok, target_to)

    results: List[str] = []
    for i in range(0, len(texts), max(1, int(batch_size))):
        chunk = [(_t or "").strip() for _t in texts[i : i + batch_size]]
        if all(not c for c in chunk):
            results.extend(["" for _ in chunk])
            continue

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


# ----------------------------
# Time-sync retiming (core fix)
# ----------------------------
def translate_segment_words_time_synced(
    source_words: List[Dict[str, Any]],
    translated_text: str,
    seg_start: float,
    seg_end: float,
    merge_contractions: bool = True,
) -> List[Dict[str, Any]]:
    """
    Given source word timestamps + translated text, output translated tokens with timestamps that:
      - stay within [seg_start, seg_end]
      - preserve order
      - split each source word’s time among the translated tokens assigned to it
    """
    src = _sanitize_words(source_words)
    tgt_tokens = _tokenize_for_timing(translated_text, merge_contractions=merge_contractions)

    if not tgt_tokens:
        return []

    if not src:
        return _uniform_time_tokens(tgt_tokens, seg_start, seg_end)

    counts = _allocate_tokens_to_source_words(src, tgt_tokens)

    out: List[Dict[str, Any]] = []
    ti = 0
    for si, sw in enumerate(src):
        n = counts[si]
        if n <= 0:
            continue

        chunk = tgt_tokens[ti : ti + n]
        ti += n

        sw_start = max(float(sw["start"]), float(seg_start))
        sw_end = min(float(sw["end"]), float(seg_end))
        if sw_end <= sw_start:
            sw_end = min(float(seg_end), sw_start + 1e-3)

        out.extend(_split_interval_across_tokens(chunk, sw_start, sw_end))

    # If any tokens remain (rare rounding edge-case), assign them at the end
    if ti < len(tgt_tokens):
        last_end = out[-1]["end"] if out else float(seg_start)
        out.extend(_split_interval_across_tokens(tgt_tokens[ti:], last_end, float(seg_end)))

    return _clamp_and_monotonic(out, float(seg_start), float(seg_end))


def retime_translation_segments(
    segments: List[Dict[str, Any]],
    target_words_key: str = "words",
    source_words_key: str = "source_words",
    merge_contractions: bool = True,
) -> List[Dict[str, Any]]:
    """
    For each segment:
      - reads seg["text"] (translated)
      - reads seg[source_words_key] (original whisper word timings)
      - writes seg[target_words_key] with retimed translated tokens
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
    Backward-compat shim: uniformly split translated tokens across the segment.
    (Your pipeline can still call this, but retime_translation_segments() is the real fix.)
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
    Returns timing tokens as *words*, with punctuation attached:
      "love, I've been..." -> ["love,", "I've", "been", "noticing", "you", "for", "days", "and", "the", "truth..."]
    No standalone tokens like "," "." "'" etc.
    """
    text = _normalize_spaces(text)
    if not text:
        return []

    # Split on whitespace first
    raw = text.split()

    out: List[str] = []
    for tok in raw:
        tok = tok.strip()
        if not tok:
            continue

        if merge_contractions:
            tok = _normalize_apostrophes(tok)
            # nothing else needed: "I've", "don't", "let's" stay intact naturally

        out.append(tok)

    return out


def _normalize_apostrophes(s: str) -> str:
    # unify curly apostrophes to straight so downstream matching is consistent
    return s.replace("’", "'")

def _merge_english_contractions(tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]

        # I + 'm => I'm
        if i + 1 < len(tokens) and re.fullmatch(r"[A-Za-z]+", t) and re.fullmatch(r"['’][A-Za-z]+", tokens[i + 1]):
            out.append(t + tokens[i + 1])
            i += 2
            continue

        # don ' t => don't  (common from some tokenizers)
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
    if re.fullmatch(r"[^\w]+", tok):
        return 0.15
    core = re.sub(r"[^\w]+", "", tok)
    return max(0.4, float(len(core)))


def _allocate_tokens_to_source_words(src_words: List[Dict[str, Any]], tgt_tokens: List[str]) -> List[int]:
    """
    Decide how many target tokens go to each source word.
    """
    S = len(src_words)
    T = len(tgt_tokens)
    if S <= 0:
        return []

    counts = [0] * S

    if T >= S:
        # start with 1 per source word
        for i in range(S):
            counts[i] = 1
        remaining = T - S

        durs = [max(1e-3, float(w["end"]) - float(w["start"])) for w in src_words]
        total = sum(durs) or 1.0

        add = [int(round(remaining * (d / total))) for d in durs]
        diff = remaining - sum(add)

        if diff != 0:
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

        # force exact match to T
        cur = sum(counts)
        if cur < T:
            counts[-1] += (T - cur)
        elif cur > T:
            extra = cur - T
            for i in range(S - 1, -1, -1):
                take = min(extra, max(0, counts[i] - 1))
                counts[i] -= take
                extra -= take
                if extra <= 0:
                    break
            i = S - 1
            while extra > 0 and i >= 0:
                take = min(extra, counts[i])
                counts[i] -= take
                extra -= take
                i -= 1

        return counts

    # T < S: give tokens to longest source words
    durs = [max(1e-3, float(w["end"]) - float(w["start"])) for w in src_words]
    order = sorted(range(S), key=lambda i: durs[i], reverse=True)
    for i in range(T):
        counts[order[i]] = 1

    # stabilize order: keep 1s but ensure left-to-right consistency
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
