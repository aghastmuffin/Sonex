import whisperx
from backbone.ltra.letra_toolkit import transcribe, align, separate
print("[pipeline] letra_toolkit inited")
print("[pipeline] separate...")
separate("test_audio/ADV.mp3")
input("Continue?")
transcribe("ADV/vocals.mp3", "vocals_whisper_segments.json", _align=True)
print("[pipeline] transcribed successfully.")
#print("[pipeline] begin alignment...")
#align("RREGATON_ExperimentoMykeTowers/vocals.mp3", "vocals_whisper_segments.json", "output.json", "es")
print("[pipeline] done.")