from __future__ import annotations

import argparse
import json
import pathlib
import time
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional, Tuple

from backbone.ltra._translation_layer import (
    translate_segment_words_time_synced,
    translate_with_alignment,
)


# -----------------------------
# Timeouts (macOS/Linux)
# -----------------------------
class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("Timed out")


def run_with_alarm(seconds: int, fn, *args, **kwargs):
    """Run fn with a hard timeout across platforms."""
    timeout_s = max(1, int(seconds))

    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGALRM, _alarm_handler)
        try:
            signal.alarm(timeout_s)
            return fn(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            raise TimeoutError("Timed out") from exc


# -----------------------------
# Main translation (OpusMT primary, NLLB fallback)
# -----------------------------
def translate_segments(
    input_segments: List[Dict[str, Any]],
    from_code: str = "es",
    to_code: str = "en",
    *,
    verbose: bool = False,
    use_opus: bool = True,
    align_timeout_s: int = 15,
    safe_align: bool = True,
    # legacy compat (ignored)
    allow_download: bool = True,
    index_timeout_s: int = 45,
    download_timeout_s: int = 900,
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

    print(f"Translating {total} segments: {from_code}->{to_code} (opus={use_opus})", flush=True)

    for idx, seg in enumerate(input_segments):
        seg_id = seg.get("id")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        source_text = seg.get("text", "") or ""

        if verbose:
            print(f"[{idx + 1}/{total}] seg id={seg_id} window={start:.2f}->{end:.2f}", flush=True)

        # Translate + extract word alignment in one pass
        t0 = time.perf_counter()
        translated_text = ""
        alignment = None

        def _do_translate():
            return translate_with_alignment(
                source_text,
                src_lang=from_code,
                tgt_lang=to_code,
                use_opus=use_opus,
            )

        try:
            if safe_align:
                translated_text, alignment = run_with_alarm(align_timeout_s, _do_translate)
            else:
                translated_text, alignment = _do_translate()
        except TimeoutError:
            if verbose:
                print(f"[{idx + 1}/{total}] translate timed out, using empty", flush=True)
            translated_text, alignment = "", None
        except Exception as exc:
            print(f"[{idx + 1}/{total}] translate error: {exc}", flush=True)
            translated_text, alignment = "", None

        t1 = time.perf_counter()
        if verbose:
            print(f"[{idx + 1}/{total}] translate+align done in {(t1 - t0):.3f}s", flush=True)

        # Apply alignment-aware timestamping
        source_words = seg.get("words", []) or []
        target_words = translate_segment_words_time_synced(
            source_words,
            translated_text,
            start,
            end,
            merge_contractions=True,
            alignment=alignment,
        )

        if verbose or (idx + 1) % 5 == 0:
            has_align = "aligned" if alignment else "positional"
            print(f"[{idx + 1}/{total}] src={len(source_words)} tgt={len(target_words)} ({has_align})", flush=True)

        target_words = _prefix_word_spaces(target_words)

        new_seg = dict(seg)
        new_seg["text"] = translated_text
        new_seg["words"] = target_words
        new_seg["source_text"] = source_text
        new_seg["source_words"] = seg.get("words", [])
        if alignment:
            new_seg["alignment"] = alignment
        translated.append(new_seg)

    return translated


def translate_file(
    input_path: pathlib.Path | str,
    *,
    output_path: pathlib.Path | str | None = None,
    from_lang: str = "es",
    to_lang: str = "en",
    verbose: bool = False,
    use_opus: bool = True,
    align_timeout_s: int = 15,
    safe_align: bool = True,
    # legacy compat
    allow_download: bool = True,
    index_timeout_s: int = 45,
    download_timeout_s: int = 900,
) -> pathlib.Path:
    in_path = pathlib.Path(input_path)
    out_path = pathlib.Path(output_path) if output_path else in_path.with_name("translated.json")

    with in_path.open("r", encoding="utf-8") as f:
        segments = json.load(f)

    translated_segments = translate_segments(
        segments,
        from_lang,
        to_lang,
        verbose=verbose,
        use_opus=use_opus,
        align_timeout_s=align_timeout_s,
        safe_align=safe_align,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(translated_segments)} segments to {out_path}", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Translate whisper segments (OpusMT/NLLB) with attention-aligned word timestamps.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input whisper segments JSON")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: translated.json)")
    parser.add_argument("--from-lang", default="es", help="Source language ISO code")
    parser.add_argument("--to-lang", default="en", help="Target language ISO code")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-opus", action="store_true", help="Force NLLB instead of OpusMT")
    parser.add_argument("--align-timeout", type=int, default=15, help="Timeout per segment (seconds)")
    parser.add_argument("--unsafe-align", action="store_true", help="Disable timeout wrapper")

    args = parser.parse_args()

    out_path = translate_file(
        args.input,
        output_path=args.output,
        from_lang=args.from_lang,
        to_lang=args.to_lang,
        verbose=args.verbose,
        use_opus=not args.no_opus,
        align_timeout_s=args.align_timeout,
        safe_align=not args.unsafe_align,
    )
    print(f"Done: {out_path}", flush=True)


if __name__ == "__main__":
    main()
