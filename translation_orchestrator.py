import _translation_layer as tl
import pathlib, json

head = "presiento"
source = "vocals_whisper_segments.json"
to = "translated_vocals_whisper_segments.json"

with open(pathlib.Path(head, source), "r") as f:
    lyricsog = json.load(f)
    f.close()

for lyric in lyricsog:
    whole_seg = lyric["text"]
    translated = tl.nllbtranslate(whole_seg, "eng_Latn", "spa_Latn")
    print(whole_seg)
    print(f"Translated: {translated}")
    print(round(lyric["id"] / len(lyricsog), 2) * 100)