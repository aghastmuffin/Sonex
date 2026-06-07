import argparse
import json
import os
import re
from typing import Any, Dict, List


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _segment_text(segment: Dict[str, Any]) -> str:
    text = str(segment.get("text", "") or "").strip()
    if text:
        return text

    words = segment.get("words", []) or []
    pieces = []
    for word in words:
        if isinstance(word, dict):
            w = str(word.get("word", "") or "").strip()
            if w:
                pieces.append(w)
    return " ".join(pieces).strip()


def _extract_lyrics(data: Any) -> str:
    segments: List[Dict[str, Any]] = []

    if isinstance(data, list):
        segments = [s for s in data if isinstance(s, dict)]
    elif isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            segments = [s for s in data["segments"] if isinstance(s, dict)]
        elif "text" in data and isinstance(data["text"], str):
            return _normalize_space(data["text"])

    parts = []
    for seg in segments:
        seg_text = _segment_text(seg)
        if seg_text:
            parts.append(seg_text)

    return _normalize_space(" ".join(parts))


def build_output_path(output_name: str) -> str:
    safe_name = os.path.basename(output_name)
    if not safe_name:
        raise ValueError("Output name must not be empty")

    out_dir = "lyrics_study/equivlyrics"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, safe_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all lyrics from a generated JSON file into one continuous string."
    )
    parser.add_argument("input_json", help="Path to input JSON file")
    parser.add_argument(
        "output_name",
        help="Output filename to create inside equivlyrics/ (example: song_01.txt)",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    lyrics = _extract_lyrics(data)
    output_path = build_output_path(args.output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(lyrics)

    print(f"Wrote {len(lyrics)} characters to {output_path}")


if __name__ == "__main__":
    main()
