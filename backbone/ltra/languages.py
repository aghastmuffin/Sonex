"""Shared language metadata and cross-platform locale helpers for Sonex."""

from __future__ import annotations

import locale
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

# Sentinel for Whisper auto-detect (propagates cleanly through CLI args).
DETECT_LANGUAGE = "detect_language"

# ISO 639-1 -> display name for UI and logging.
LANGUAGE_NAMES: Dict[str, str] = {
    "ar": "Arabic",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

# MFA acoustic/dictionary model names keyed by ISO code.
MFA_LANGUAGE_NAMES: Dict[str, str] = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "zh": "mandarin",
    "ja": "japanese",
    "ko": "korean",
    "ar": "arabic",
    "hi": "hindi",
    "bn": "bengali",
    "pa": "punjabi",
    "tr": "turkish",
    "vi": "vietnamese",
    "pl": "polish",
    "nl": "dutch",
    "sv": "swedish",
    "no": "norwegian",
    "da": "danish",
    "fi": "finnish",
    "he": "hebrew",
    "el": "greek",
    "th": "thai",
    "id": "indonesian",
    "uk": "ukrainian",
    "cs": "czech",
    "ro": "romanian",
    "hu": "hungarian",
    "ca": "catalan",
    "hr": "croatian",
    "ms": "malay",
    "tl": "tagalog",
}

TRANSLATION_MODES: List[Tuple[str, str]] = [
    ("None (keep source)", "none"),
    ("OpusMT / NLLB (to selected language)", "argos"),
    ("Whisper (to English only)", "whisper"),
    ("Both (Whisper + OpusMT/NLLB)", "both"),
]


def sorted_language_items(include_detect: bool = False) -> List[Tuple[str, str]]:
    """Return (display_name, iso_code) pairs sorted alphabetically by display name."""
    items = [(name, code) for code, name in LANGUAGE_NAMES.items()]
    items.sort(key=lambda pair: pair[0].lower())
    if include_detect:
        return [("Detect language", DETECT_LANGUAGE)] + items
    return items


def normalize_lang_code(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    code = str(code).strip().lower()
    if not code or code in {"detect", "detect_language", "auto", "auto-detect"}:
        return None
    if code in LANGUAGE_NAMES:
        return code
    for iso, name in LANGUAGE_NAMES.items():
        if name.lower() == code:
            return iso
    for iso, name in MFA_LANGUAGE_NAMES.items():
        if name.lower() == code:
            return iso
    return None


def _locale_candidates() -> Iterable[str]:
    getters = (
        lambda: locale.getlocale()[0],
        lambda: locale.getdefaultlocale()[0],
    )
    for getter in getters:
        try:
            value = getter()
        except Exception:
            value = None
        if value:
            yield str(value)

    for key in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = os.environ.get(key)
        if value and value not in {"C", "POSIX"}:
            yield value.split(".")[0]

    if sys.platform.startswith("darwin"):
        try:
            import subprocess

            out = subprocess.check_output(
                ["defaults", "read", "-g", "AppleLocale"],
                text=True,
                timeout=1,
            ).strip()
            if out:
                yield out
        except Exception:
            pass

    if sys.platform.startswith("win"):
        try:
            import ctypes

            lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            mapped = locale.windows_locale.get(lang_id)
            if mapped:
                yield mapped
        except Exception:
            pass


def get_system_language_code(fallback: str = "en") -> str:
    """Resolve the user's UI language to a supported ISO 639-1 code on any OS."""
    for candidate in _locale_candidates():
        token = str(candidate).replace("_", "-").split("-")[0].lower()
        normalized = normalize_lang_code(token)
        if normalized:
            return normalized
    normalized_fallback = normalize_lang_code(fallback) or "en"
    return normalized_fallback


def resolve_source_language(selected: Optional[str]) -> Optional[str]:
    """Return None when the user wants Whisper auto-detect."""
    return normalize_lang_code(selected)


def resolve_target_language(selected: Optional[str], fallback: Optional[str] = None) -> str:
    """Target language defaults to the system language when not explicitly chosen."""
    normalized = normalize_lang_code(selected)
    if normalized:
        return normalized
    return get_system_language_code(fallback or "en")
