# _translation_layer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer

# ----------------------------
# Global singleton caches
# ----------------------------
_NLLB_TOKENIZER = None
_NLLB_MODEL = None
_NLLB_DEVICE = None

_OPUS_TOKENIZER = None
_OPUS_MODEL = None
_OPUS_DEVICE = None
_OPUS_PAIR = None  # (src, tgt) currently loaded

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ----------------------------
# Device + model loading
# ----------------------------
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_opus(src_lang: str, tgt_lang: str):
    """Load Helsinki-NLP/OpusMT model for a specific language pair."""
    global _OPUS_TOKENIZER, _OPUS_MODEL, _OPUS_DEVICE, _OPUS_PAIR

    pair = (src_lang, tgt_lang)
    if _OPUS_MODEL is not None and _OPUS_PAIR == pair:
        return _OPUS_TOKENIZER, _OPUS_MODEL, _OPUS_DEVICE

    device = _get_device()
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    tok = MarianTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = MarianMTModel.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager")
    model.to(device)
    model.eval()

    _OPUS_TOKENIZER = tok
    _OPUS_MODEL = model
    _OPUS_DEVICE = device
    _OPUS_PAIR = pair
    return tok, model, device


def _load_nllb(model_name: str = "facebook/nllb-200-distilled-600M"):
    global _NLLB_TOKENIZER, _NLLB_MODEL, _NLLB_DEVICE

    if _NLLB_MODEL is not None and _NLLB_TOKENIZER is not None and _NLLB_DEVICE is not None:
        return _NLLB_TOKENIZER, _NLLB_MODEL, _NLLB_DEVICE

    device = _get_device()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager")
    model.to(device)
    model.eval()

    _NLLB_TOKENIZER = tok
    _NLLB_MODEL = model
    _NLLB_DEVICE = device
    return tok, model, device


# ----------------------------
# Subword → word index mapping
# ----------------------------
def _map_subwords_to_words(tokenizer, text: str, token_ids: List[int]) -> List[int]:
    """
    For each subword token in token_ids, return which whitespace-word index it belongs to.
    Special tokens get -1.
    """
    words = text.split()
    if not words:
        return [-1] * len(token_ids)

    word_map = []
    current_word_idx = 0
    chars_consumed_in_word = 0

    for tid in token_ids:
        if tid in tokenizer.all_special_ids:
            word_map.append(-1)
            continue

        dec = tokenizer.decode([tid], skip_special_tokens=False)
        clean = dec.replace("\u2581", "").replace("\u0120", "").strip()
        if not clean:
            word_map.append(min(current_word_idx, len(words) - 1))
            continue

        is_word_start = dec.startswith("\u2581") or dec.startswith("\u0120") or dec.startswith(" ")
        if is_word_start and chars_consumed_in_word > 0:
            current_word_idx += 1
            chars_consumed_in_word = 0

        safe_idx = min(current_word_idx, len(words) - 1)
        word_map.append(safe_idx)
        chars_consumed_in_word += len(clean)

    return word_map


def _map_generated_to_words(tokenizer, generated_ids: List[int], translated_text: str) -> List[int]:
    """Map generated subword token indices to target word indices."""
    words = translated_text.split()
    if not words or not generated_ids:
        return [-1] * len(generated_ids)

    word_map = []
    current_word_idx = 0
    chars_in_current = 0

    for tid in generated_ids:
        if tid in tokenizer.all_special_ids:
            word_map.append(-1)
            continue

        piece = tokenizer.decode([tid], skip_special_tokens=True)
        if not piece:
            word_map.append(max(0, min(current_word_idx, len(words) - 1)))
            continue

        if piece.startswith(" ") and chars_in_current > 0:
            current_word_idx += 1
            chars_in_current = 0

        safe_idx = min(current_word_idx, len(words) - 1)
        word_map.append(safe_idx)
        chars_in_current += len(piece.strip())

    return word_map


# ----------------------------
# Cross-attention alignment extraction
# ----------------------------
def _extract_word_alignment(
    cross_attentions,
    src_word_map: List[int],
    tgt_word_map: List[int],
    num_src_words: int,
    num_tgt_words: int,
) -> List[int]:
    """
    From cross-attention weights, build a target_word -> source_word alignment.
    Returns: for each target word index, the source word index it most attends to.
    """
    if num_tgt_words == 0 or num_src_words == 0:
        return []

    tgt_to_src_scores = np.zeros((num_tgt_words, num_src_words), dtype=np.float64)

    for step_idx, step_attns in enumerate(cross_attentions):
        if step_idx >= len(tgt_word_map):
            break
        tgt_widx = tgt_word_map[step_idx]
        if tgt_widx < 0:
            continue

        # Average across layers and heads for this step
        layer_attns = []
        for layer_attn in step_attns:
            if layer_attn is None:
                continue
            # (1, heads, cur_tgt_len, src_len) -> take last tgt position
            attn = layer_attn[0, :, -1, :].float().cpu().numpy()
            layer_attns.append(attn.mean(axis=0))

        if not layer_attns:
            continue

        avg_attn = np.mean(layer_attns, axis=0)

        for src_subword_idx, score in enumerate(avg_attn):
            if src_subword_idx >= len(src_word_map):
                break
            src_widx = src_word_map[src_subword_idx]
            if src_widx < 0:
                continue
            tgt_to_src_scores[tgt_widx, src_widx] += score

    alignment = []
    for tw in range(num_tgt_words):
        row = tgt_to_src_scores[tw]
        if row.sum() < 1e-9:
            alignment.append(min(tw, num_src_words - 1))
        else:
            alignment.append(int(np.argmax(row)))

    return alignment


# ----------------------------
# Translation with alignment
# ----------------------------
def translate_with_alignment(
    text: str,
    src_lang: str,
    tgt_lang: str,
    *,
    use_opus: bool = True,
    max_new_tokens: int = 256,
    num_beams: int = 1,
) -> Tuple[str, Optional[List[int]]]:
    """
    Translate text and return (translated_text, word_alignment).
    word_alignment[i] = source word index that target word i aligns to.

    Uses greedy decoding (num_beams=1) to get coherent cross-attention.
    Beam search scatters attention across hypotheses making it unreliable for alignment.
    """
    text = (text or "").strip()
    if not text:
        return "", None

    src_words = text.split()
    num_src_words = len(src_words)

    tok = model = device = None
    is_opus = False

    if use_opus:
        try:
            tok, model, device = _load_opus(src_lang, tgt_lang)
            is_opus = True
        except Exception:
            pass

    if tok is None:
        tok, model, device = _load_nllb()
        tok.src_lang = _nllb_lang_code(src_lang)
        is_opus = False

    inputs = tok([text], return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs["input_ids"][0].tolist()
    src_word_map = _map_subwords_to_words(tok, text, input_ids)

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        output_attentions=True,
        return_dict_in_generate=True,
    )
    if not is_opus:
        forced_id = _resolve_lang_token_id(tok, _nllb_lang_code(tgt_lang))
        gen_kwargs["forced_bos_token_id"] = forced_id

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    generated_ids = out.sequences[0].tolist()
    translated_text = _normalize_spaces(tok.decode(generated_ids, skip_special_tokens=True))
    num_tgt_words = len(translated_text.split())

    alignment = None
    if hasattr(out, "cross_attentions") and out.cross_attentions:
        try:
            gen_no_special = [t for t in generated_ids if t not in tok.all_special_ids]
            tgt_word_map = _map_generated_to_words(tok, gen_no_special, translated_text)

            alignment = _extract_word_alignment(
                out.cross_attentions,
                src_word_map,
                tgt_word_map,
                num_src_words,
                num_tgt_words,
            )
        except Exception:
            alignment = None

    return translated_text, alignment


# ----------------------------
# NLLB language code mapping
# ----------------------------
_NLLB_CODES = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab",
    "hi": "hin_Deva", "tr": "tur_Latn", "pl": "pol_Latn", "sv": "swe_Latn",
    "da": "dan_Latn", "fi": "fin_Latn", "no": "nob_Latn", "uk": "ukr_Cyrl",
    "cs": "ces_Latn", "ro": "ron_Latn", "hu": "hun_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "th": "tha_Thai", "vi": "vie_Latn", "id": "ind_Latn",
    "ms": "zsm_Latn", "tl": "tgl_Latn", "ca": "cat_Latn", "hr": "hrv_Latn",
    "bn": "ben_Beng", "pa": "pan_Guru",
}


def _nllb_lang_code(iso: str) -> str:
    """Convert ISO 639-1 code to NLLB format. Pass through if already NLLB format."""
    if "_" in iso and len(iso) > 3:
        return iso
    return _NLLB_CODES.get(iso, iso)


# ----------------------------
# Legacy translation API (backwards-compatible)
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
# Time-sync retiming (alignment-aware)
# ----------------------------
def translate_segment_words_time_synced(
    source_words: List[Dict[str, Any]],
    translated_text: str,
    seg_start: float,
    seg_end: float,
    merge_contractions: bool = True,
    alignment: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Given source word timestamps + translated text, output translated tokens with timestamps.

    If `alignment` is provided (target_word_idx -> source_word_idx from attention extraction),
    each target word inherits the time window of its aligned source word. Multiple target words
    aligned to the same source word split that source word's interval.

    Without alignment, falls back to proportional left-to-right distribution.
    """
    src = _sanitize_words(source_words)
    tgt_tokens = _tokenize_for_timing(translated_text, merge_contractions=merge_contractions)

    if not tgt_tokens:
        return []

    if not src:
        return _uniform_time_tokens(tgt_tokens, seg_start, seg_end)

    if alignment and len(alignment) == len(tgt_tokens):
        return _alignment_based_timing(src, tgt_tokens, alignment, seg_start, seg_end)

    # Fallback: positional allocation (legacy behavior)
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

    if ti < len(tgt_tokens):
        last_end = out[-1]["end"] if out else float(seg_start)
        out.extend(_split_interval_across_tokens(tgt_tokens[ti:], last_end, float(seg_end)))

    return _clamp_and_monotonic(out, float(seg_start), float(seg_end))


def _alignment_based_timing(
    src_words: List[Dict[str, Any]],
    tgt_tokens: List[str],
    alignment: List[int],
    seg_start: float,
    seg_end: float,
) -> List[Dict[str, Any]]:
    """
    Use semantic alignment to assign timestamps. Each target token inherits the timing
    of its aligned source word. When multiple targets align to the same source word,
    that source word's interval is subdivided.

    Maintains monotonic output for subtitle display by interpolating when alignment
    would otherwise cause backward time jumps (reordering).
    """
    S = len(src_words)
    T = len(tgt_tokens)

    safe_alignment = [max(0, min(a, S - 1)) for a in alignment]

    # Group target tokens by which source word they align to
    groups: Dict[int, List[int]] = {}
    for ti, si in enumerate(safe_alignment):
        groups.setdefault(si, []).append(ti)

    # Assign raw time for each target token based on its aligned source word
    raw_times: List[Tuple[float, float]] = [(0.0, 0.0)] * T
    for si, ti_list in groups.items():
        sw = src_words[si]
        sw_start = max(float(sw["start"]), seg_start)
        sw_end = min(float(sw["end"]), seg_end)
        if sw_end <= sw_start:
            sw_end = sw_start + 1e-3

        n = len(ti_list)
        weights = [_token_weight(tgt_tokens[ti]) for ti in ti_list]
        total_w = sum(weights) or 1.0
        dur = sw_end - sw_start

        t = sw_start
        for j, ti in enumerate(ti_list):
            frac = weights[j] / total_w
            dt = dur * frac
            raw_times[ti] = (t, t + dt if j < n - 1 else sw_end)
            t = raw_times[ti][1]

    # Enforce monotonicity: target tokens must appear in time order for display.
    out: List[Dict[str, Any]] = []
    prev_end = seg_start

    for ti in range(T):
        s, e = raw_times[ti]

        if s < prev_end:
            s = prev_end
        if e <= s:
            next_natural = seg_end
            for future_ti in range(ti + 1, T):
                fs, _ = raw_times[future_ti]
                if fs > s:
                    next_natural = fs
                    break
            remaining_tokens = T - ti
            available = next_natural - s
            e = s + max(1e-3, available / remaining_tokens)

        e = min(e, seg_end)
        if e <= s:
            e = s + 1e-3

        out.append({"word": tgt_tokens[ti], "start": s, "end": e})
        prev_end = e

    return _clamp_and_monotonic(out, seg_start, seg_end)


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
        alignment = seg2.get("alignment", None)
        seg2[target_words_key] = translate_segment_words_time_synced(
            src_words,
            translated_text,
            seg_start,
            seg_end,
            merge_contractions=merge_contractions,
            alignment=alignment,
        )
        out.append(seg2)
    return out


def segment_text_to_words(text: str, start: float, end: float) -> List[Dict[str, Any]]:
    """Backward-compat shim: uniformly split translated tokens across the segment."""
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
    Returns timing tokens as *words*, with punctuation attached.
    No standalone tokens like "," "." "'" etc.
    """
    text = _normalize_spaces(text)
    if not text:
        return []

    raw = text.split()

    out: List[str] = []
    for tok in raw:
        tok = tok.strip()
        if not tok:
            continue

        if merge_contractions:
            tok = _normalize_apostrophes(tok)

        out.append(tok)

    return out


def _normalize_apostrophes(s: str) -> str:
    return s.replace("\u2019", "'")


def _merge_english_contractions(tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]

        if i + 1 < len(tokens) and re.fullmatch(r"[A-Za-z]+", t) and re.fullmatch(r"['\u2019][A-Za-z]+", tokens[i + 1]):
            out.append(t + tokens[i + 1])
            i += 2
            continue

        if i + 2 < len(tokens) and re.fullmatch(r"[A-Za-z]+", t) and tokens[i + 1] in ["'", "\u2019"] and re.fullmatch(r"[A-Za-z]+", tokens[i + 2]):
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
    Decide how many target tokens go to each source word (positional, duration-proportional).
    Used as fallback when no alignment data is available.
    """
    S = len(src_words)
    T = len(tgt_tokens)
    if S <= 0:
        return []

    counts = [0] * S

    if T >= S:
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
