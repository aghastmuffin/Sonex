from __future__ import annotations

import argparse
import json
import pathlib
import time
import signal
from typing import Any, Dict, List, Tuple

import argostranslate.package
import argostranslate.translate

from backbone.ltra._translation_layer import translate_segment_words_time_synced


# -----------------------------
# Timeouts (macOS/Linux)
# -----------------------------
class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("Timed out")


def run_with_alarm(seconds: int, fn, *args, **kwargs):
    """Run fn with SIGALRM timeout (works on macOS/Linux main thread)."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(max(1, int(seconds)))
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# -----------------------------
# Argos model loading/install
# -----------------------------
def ensure_model_once(
    from_code: str,
    to_code: str,
    *,
    verbose: bool = False,
    allow_download: bool = True,
    index_timeout_s: int = 45,
    download_timeout_s: int = 900,
) -> Tuple[Any, Any, Any]:
    """
    Ensure Argos package exists and return (from_lang, to_lang, translation).
    Adds hard timeouts so network can’t hang forever.
    """
    installed = argostranslate.translate.get_installed_languages()
    if verbose:
        installed_codes = [f"{l.code}:{l.name}" for l in installed]
        print(f"Installed languages: {installed_codes}", flush=True)

    # Fast path: already installed
    try:
        from_lang = next(l for l in installed if l.code == from_code)
        to_lang = next(l for l in installed if l.code == to_code)
        translation = from_lang.get_translation(to_lang)
        if verbose:
            print(f"Using installed Argos package {from_code}->{to_code}", flush=True)
        return from_lang, to_lang, translation
    except Exception as exc:
        if verbose:
            print(f"Argos package {from_code}->{to_code} not installed ({exc})", flush=True)

    if not allow_download:
        raise RuntimeError(
            f"Argos package {from_code}->{to_code} not installed and downloads are disabled"
        )

    # Download/install path
    if verbose:
        print(f"Fetching package index for {from_code}->{to_code} (timeout {index_timeout_s}s)...", flush=True)

    t0 = time.perf_counter()
    run_with_alarm(index_timeout_s, argostranslate.package.update_package_index)
    t1 = time.perf_counter()

    available_packages = argostranslate.package.get_available_packages()
    try:
        pkg = next(p for p in available_packages if p.from_code == from_code and p.to_code == to_code)
    except StopIteration:
        raise RuntimeError(f"No Argos package found for {from_code}->{to_code}")

    if verbose:
        print(f"Index fetched in {(t1 - t0):.2f}s", flush=True)
        print(f"Downloading {pkg} (timeout {download_timeout_s}s)...", flush=True)

    t2 = time.perf_counter()
    # pkg.download() does the HTTPS fetch; wrap it too.
    pkg_path = run_with_alarm(download_timeout_s, pkg.download)
    argostranslate.package.install_from_path(pkg_path)
    t3 = time.perf_counter()

    if verbose:
        print(f"Download/install complete in {(t3 - t2):.2f}s", flush=True)

    installed = argostranslate.translate.get_installed_languages()
    from_lang = next(l for l in installed if l.code == from_code)
    to_lang = next(l for l in installed if l.code == to_code)
    translation = from_lang.get_translation(to_lang)
    return from_lang, to_lang, translation


# -----------------------------
# Optional align timeout + fallback
# -----------------------------
def align_words_time_synced_safe(
    source_words: List[Dict[str, Any]],
    translated_text: str,
    start: float,
    end: float,
    *,
    timeout_s: int = 8,
    merge_contractions: bool = True,
) -> List[Dict[str, Any]]:
    """
    Wrap aligner in an alarm timeout. If it times out, fall back to a simple even-spread timing.
    """
    def _do_align():
        return translate_segment_words_time_synced(
            source_words,
            translated_text,
            start,
            end,
            merge_contractions=merge_contractions,
        )

    try:
        return run_with_alarm(timeout_s, _do_align)
    except TimeoutError:
        # Fallback: naive timings over translated tokens
        toks = translated_text.split()
        if not toks:
            return []
        dur = max(0.0, float(end) - float(start))
        step = dur / max(1, len(toks))
        out = []
        t = float(start)
        for tok in toks:
            out.append({"word": tok, "start": t, "end": t + step})
            t += step
        return out


# -----------------------------
# Main translation
# -----------------------------
def translate_segments(
    input_segments: List[Dict[str, Any]],
    from_code: str = "es",
    to_code: str = "en",
    *,
    verbose: bool = False,
    allow_download: bool = True,
    index_timeout_s: int = 45,
    download_timeout_s: int = 900,
    align_timeout_s: int = 8,
    safe_align: bool = True,
) -> List[Dict[str, Any]]:
    translated: List[Dict[str, Any]] = []
    total = len(input_segments)

    def _prefix_word_spaces(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for idx, w in enumerate(words):
            nw = dict(w)
            text = str(nw.get("word", ""))
            if idx > 0:
                text = f" {text}"
            nw["word"] = text
            out.append(nw)
        return out

    # IMPORTANT: ensure model once
    _, _, translation = ensure_model_once(
        from_code,
        to_code,
        verbose=verbose,
        allow_download=allow_download,
        index_timeout_s=index_timeout_s,
        download_timeout_s=download_timeout_s,
    )

    for idx, seg in enumerate(input_segments):
        seg_id = seg.get("id")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        print(f"[{idx + 1}/{total}] start seg id={seg_id} window={start}->{end}", flush=True)

        source_text = seg.get("text", "") or ""

        t0 = time.perf_counter()
        translated_text = translation.translate(source_text).strip() if source_text else ""
        t1 = time.perf_counter()
        if verbose:
            print(f"[{idx + 1}/{total}] translate done in {(t1 - t0):.3f}s", flush=True)

        t2 = time.perf_counter()
        if safe_align:
            target_words = align_words_time_synced_safe(
                seg.get("words", []) or [],
                translated_text,
                start,
                end,
                timeout_s=align_timeout_s,
                merge_contractions=True,
            )
        else:
            target_words = translate_segment_words_time_synced(
                seg.get("words", []) or [],
                translated_text,
                start,
                end,
                merge_contractions=True,
            )
        t3 = time.perf_counter()
        if verbose:
            print(f"[{idx + 1}/{total}] align done in {(t3 - t2):.3f}s", flush=True)

        print(f"[{idx + 1}/{total}] words: src={len(seg.get('words', []) or [])} tgt={len(target_words)}", flush=True)

        target_words = _prefix_word_spaces(target_words)

        new_seg = dict(seg)
        new_seg["text"] = translated_text
        new_seg["words"] = target_words
        new_seg["source_text"] = source_text
        new_seg["source_words"] = seg.get("words", [])
        translated.append(new_seg)

    return translated


def translate_file(
    input_path: pathlib.Path | str,
    *,
    output_path: pathlib.Path | str | None = None,
    from_lang: str = "es",
    to_lang: str = "en",
    verbose: bool = False,
    allow_download: bool = True,
    index_timeout_s: int = 45,
    download_timeout_s: int = 900,
    align_timeout_s: int = 8,
    safe_align: bool = True,
) -> pathlib.Path:
    in_path = pathlib.Path(input_path)
    out_path = pathlib.Path(output_path) if output_path else in_path.with_name("argos_translated.json")

    with in_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    print("calling translated segments", flush=True)

    translated_segments = translate_segments(
        segments,
        from_lang,
        to_lang,
        verbose=verbose,
        allow_download=allow_download,
        index_timeout_s=index_timeout_s,
        download_timeout_s=download_timeout_s,
        align_timeout_s=align_timeout_s,
        safe_align=safe_align,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"Wrote {len(translated_segments)} segments to {out_path}", flush=True)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Translate whisper segments with Argos Translate and write argos_translated.json (safe timeouts).",
    )
    parser.add_argument("-i", "--input", default="presiento_mfa/vocals_whisper_segments.json")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--from-lang", default="es")
    parser.add_argument("--to-lang", default="en")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--index-timeout", type=int, default=45)
    parser.add_argument("--download-timeout", type=int, default=900)
    parser.add_argument("--align-timeout", type=int, default=8)
    parser.add_argument("--unsafe-align", action="store_true", help="Disable align timeout/fallback")

    args = parser.parse_args()

    out_path = translate_file(
        args.input,
        output_path=args.output,
        from_lang=args.from_lang,
        to_lang=args.to_lang,
        verbose=args.verbose,
        allow_download=not args.skip_install,
        index_timeout_s=args.index_timeout,
        download_timeout_s=args.download_timeout,
        align_timeout_s=args.align_timeout,
        safe_align=not args.unsafe_align,
    )
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()