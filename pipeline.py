import whisperx
from backbone.ltra.letra_toolkit import transcribe, align

print("letra_toolkit inited")
align("RREGATON_ExperimentoMykeTowers/vocals.mp3", "vocals_whisper_segments.json", "output.json", "es")
print("done.")