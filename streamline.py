import whisperx, os
print("initializing the lyric analysis library")
from backbone.ltra.letra_toolkit import transcribe, align, separate
SUBFOLDER = "test_audio"
AUDIO = "RREGATON_ExperimentoMykeTowers.mp3"
LANG = "en"
OUT = "output"
print("splitting")
separate(SUBFOLDER + "/" + AUDIO)
print("transcribing")
transcribe(f"{AUDIO.removesuffix('.mp3')}/vocals.mp3", f"{OUT}/vocals_whisper_segments.json", language=LANG, _align=True)
print(f"lyric analysis library completed required job: {AUDIO}")

#os.system(f"python test_audio/gui_player.py --audio {SUBFOLDER}/{AUDIO} --align {OUT}/vocals_whisper_segments_aligned.json")