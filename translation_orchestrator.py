import _translation_layer as tl
import pathlib, json

head = "presiento"
source = "vocals_whisper_segments.json"
to = "translated_vocals_whisper_segments.json"

in_path = pathlib.Path(head, source)
out_path = pathlib.Path(head, to)

with open(in_path, "r", encoding="utf-8") as f:
    lyricsog = json.load(f)

translated_segments = []
for i, lyric in enumerate(lyricsog):
    whole_seg = lyric.get("text", "")

    translated = tl.nllbtranslate(whole_seg, "eng_Latn", "spa_Latn")

    # Build target word tokens (keep your existing token/timing behavior)
    # If your tl layer already returns word-level tokens, replace this call accordingly.
    target_words = tl.segment_text_to_words(translated, start=lyric["start"], end=lyric["end"])

    seg = dict(lyric)
    seg["text"] = translated
    seg["words"] = target_words

    # Preserve source words so we can re-time faithfully
    seg["source_text"] = whole_seg
    seg["source_words"] = lyric.get("words", [])

    translated_segments.append(seg)

    print(whole_seg)
    print(f"Translated: {translated}")
    print(f"{round((i+1) / len(lyricsog) * 100, 1)}%")

# Re-time translated words to match source word boundaries
translated_segments = tl.retime_translation_segments(
    translated_segments,
    target_words_key="words",
    source_words_key="source_words",
    merge_contractions=True,
)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(translated_segments, f, ensure_ascii=False, indent=2)

print(f"\nWrote: {out_path}")