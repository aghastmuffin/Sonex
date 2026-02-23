from __future__ import annotations

from typing import Optional
import warnings
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Global cache for model and tokenizer to avoid redownloading
_model_cache = {}
_tokenizer_cache = {}
_translator_cache = {}

def atranslate(text):
    import argostranslate as argos
    raise NotImplementedError("You are attempting to use a function not implemented until SONEX VER 1.3 or your buildtype doesn't specify access to test ARGOS. View _translation_layer.py. All code inside of RMO is intellectual property of Levi Taisun Kim Brown, who grants ownership of intellectual rights to Firstline LLC, a registered limited liability corporation of the great state of california. Redistribution or deobfuscation is expressly not allowed under any conditions and may be persecuted under the full extent of the law by firstline llc or any of its subsidiaries.")
def nllbtranslate(text:str, target_to:str, target_from:str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    flores_200_codes = [
        "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab", "aka_Latn",
        "als_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "arb_Latn", "ars_Arab", "ary_Arab", "arz_Arab",
        "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn",
        "ban_Latn", "bel_Cyrl", "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt",
        "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab",
        "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek",
        "eng_Latn", "epo_Latn", "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "fij_Latn", "fin_Latn",
        "fon_Latn", "fra_Latn", "fur_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn", "gle_Latn", "glg_Latn",
        "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn",
        "hun_Latn", "hye_Armn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn",
        "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor",
        "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khk_Cyrl", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl",
        "kmb_Latn", "kmr_Latn", "knc_Arab", "knc_Latn", "kon_Latn", "kor_Hang", "lao_Laoo", "lij_Latn",
        "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn",
        "luo_Latn", "lus_Latn", "lvs_Latn", "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Arab",
        "min_Latn", "mkd_Cyrl", "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn", "mya_Mymr", "nld_Latn",
        "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn", "oci_Latn", "ory_Orya",
        "pag_Latn", "pan_Guru", "pap_Latn", "pbt_Arab", "pes_Arab", "plt_Latn", "pol_Latn", "por_Latn",
        "prs_Arab", "quy_Latn", "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Olck",
        "scn_Latn", "shn_Mymr", "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab",
        "som_Latn", "sot_Latn", "spa_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn",
        "swh_Latn", "szl_Latn", "tam_Taml", "taq_Latn", "taq_Tfng", "tat_Cyrl", "tel_Telu", "tgk_Cyrl",
        "tgl_Latn", "tha_Thai", "tir_Ethi", "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn",
        "tur_Latn", "twi_Latn", "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn",
        "vec_Latn", "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn", "yue_Hant",
        "zho_Hans", "zho_Hant", "zsm_Latn", "zul_Latn"
    ]
    if target_from in flores_200_codes and target_to in flores_200_codes:
        if len(text) >= 360:
            warnings.warn("Approaching Character Limit: Text length exceeds recommended amount. This program uses a model best suited for 360 chars or less, recommended at 400.")
        checkpoint = "facebook/nllb-200-distilled-600M"
        
        # Create cache key for this language pair
        cache_key = f"{checkpoint}_{target_from}_{target_to}"
        
        # Check if translator is already cached
        if cache_key not in _translator_cache:
            print(f"Loading model {checkpoint}... (first time, will be cached)")
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            translator = pipeline('translation', model = model, tokenizer = tokenizer, src_lang = target_from, tgt_lang = target_to, max_length = 400)
            _translator_cache[cache_key] = translator
        else:
            print(f"Using cached translator for {target_from} -> {target_to}")
        
        translator = _translator_cache[cache_key]
        output = translator(text)
        return output[0]['translation_text']
    else:
        raise ValueError("Source or target langauge not in correct flores-200 format. Learn more about this here: https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200")

def calcperplex(text):
    raise NotImplementedError("Not implemented yet. No timeline avaliable. 1/22/26 10:09pm PST")


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def _tokenize_words_and_punct(s: str) -> List[str]:
    """
    Returns tokens like: ["Amor", ",", "yo", "varios", "días", ...]
    (word chunks + punctuation as separate tokens)
    """
    s = s.strip()
    if not s:
        return []
    return _WORD_RE.findall(s)

def _weights(tokens: List[str]) -> List[int]:
    """
    Weight words more than punctuation so timing doesn't get dominated by commas.
    """
    out = []
    for t in tokens:
        if re.match(r"^\w+$", t, re.UNICODE):
            out.append(max(1, len(t)))
        else:
            out.append(1)  # punctuation
    return out

def _assign_token_times(start: float, end: float, tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Assigns each token a [start,end] inside the segment by proportional weights.
    """
    dur = max(0.0, float(end) - float(start))
    if not tokens:
        return []

    w = _weights(tokens)
    total = sum(w) or 1

    out = []
    cur = float(start)
    for i, (tok, wi) in enumerate(zip(tokens, w)):
        # Last token ends exactly at segment end to avoid drift
        if i == len(tokens) - 1:
            tok_start, tok_end = cur, float(end)
        else:
            tok_dur = dur * (wi / total)
            tok_start, tok_end = cur, cur + tok_dur
        out.append({"word": tok, "start": round(tok_start, 3), "end": round(tok_end, 3)})
        cur = tok_end
    return out

def align_translated_text(
    in_json: str | Path,
    out_json: str | Path = "translated.json",
    target_from: str = "eng_Latn",
    target_to: str = "spa_Latn",
    *,
    keep_original_words: bool = False,
) -> List[Dict[str, Any]]:
    """
    Reads a time-aligned transcript JSON (segments with start/end + words),
    translates each segment text, and writes a translated.json with time-synced
    translated word timestamps.

    - keep_original_words=False: output 'words' are translated tokens with new timings
    - keep_original_words=True : output also includes 'source_words' with original timings
    """
    in_json = Path(in_json)
    out_json = Path(out_json)

    segments: List[Dict[str, Any]] = json.loads(in_json.read_text(encoding="utf-8"))

    translated_segments: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = seg.get("id")
        start = float(seg["start"])
        end = float(seg["end"])
        src_text = seg.get("text", "")

        # Translate the whole segment text (best MT quality)
        tgt_text = nllbtranslate(src_text, target_from=target_from, target_to=target_to).strip()

        # Build translated word timestamps within the same [start,end]
        tgt_tokens = _tokenize_words_and_punct(tgt_text)
        tgt_words = _assign_token_times(start, end, tgt_tokens)

        out_seg: Dict[str, Any] = {
            "id": seg_id,
            "text": tgt_text,
            "start": round(start, 3),
            "end": round(end, 3),
            "words": tgt_words,
        }

        if keep_original_words:
            out_seg["source_text"] = src_text
            out_seg["source_words"] = seg.get("words", [])

        translated_segments.append(out_seg)

    out_json.write_text(
        json.dumps(translated_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return translated_segments

# Example:
# align_translated_text("vocals_whisper_segments.json", "translated.json",
#                       target_from="spa_Latn", target_to="eng_Latn",
#                       keep_original_words=True)


if __name__ == "__main__":
    #print(nllbtranslate("The mitochondria is the powerhouse of the cell. Hello, Levi Brown!", target_from = "eng_Latn", target_to = "spa_Latn"))
    align_translated_text("vocals_whisper_segments.json", "translated.json",
                           target_from="spa_Latn", target_to="eng_Latn",
                           keep_original_words=True)
