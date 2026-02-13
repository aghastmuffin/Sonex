import os
print("initializing the lyric analysis library")
input("it's best to close all other open programs on the system. Please do so and then press [Enter]:")
from backbone.ltra.letra_toolkit import transcribe, align, separate
SUBFOLDER = "test_audio"
AUDIO = "multilingual_holding_hero_Polina.mp3"
LANG = "ru"
OUT = "output"

print("splitting")
separate(SUBFOLDER + "/" + AUDIO)
print("transcribing")
os.makedirs(OUT, exist_ok=True)
transcribe(f"{AUDIO.removesuffix('.mp3')}/vocals.mp3", f"{OUT}/vocals_whisper_segments.json", language=LANG, _align=True)
print(f"lyric analysis library completed required job: {AUDIO}")
print("handoff to beat analyzers")

print("beat handler implementation incomplete, skipped")

print("[beat handler]: Job done with beat analyzer")

#os.system(f"python test_audio/gui_player.py --audio {SUBFOLDER}/{AUDIO} --align {OUT}/vocals_whisper_segments_aligned.json")