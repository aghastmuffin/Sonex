import whisperx, os, argparse
from backbone.ltra.letra_toolkit import transcribe, align, separate

def main():
    parser = argparse.ArgumentParser(description="Sonex pipeline for audio transcription and alignment")
    parser.add_argument("--subfolder", default="test_audio", help="Subfolder containing audio files (default: test_audio)")
    parser.add_argument("--audio", default="RREGATON_ExperimentoMykeTowers.mp3", help="Audio file name (default: RREGATON_ExperimentoMykeTowers.mp3)")
    parser.add_argument("--lang", default="en", help="Language code for transcription (default: en)")
    parser.add_argument("--runfree", default=False, action='store_true', help="Run without user prompts (default: False)")

    args = parser.parse_args()
    
    SUBFOLDER = args.subfolder
    AUDIO = args.audio
    LANG = args.lang
    RUNFREE = args.runfree
    print("[pipeline] letra_toolkit inited")
    print("[pipeline] separate...")
    separate(SUBFOLDER + "/" + AUDIO)
    if not RUNFREE:
        input("Continue?")
    transcribe(f"{AUDIO.removesuffix('.mp3')}/vocals.mp3", "vocals_whisper_segments.json", language=LANG, _align=True)
    print("[pipeline] transcribed successfully.")
    #print("[pipeline] begin alignment...")
    align(f"{AUDIO.removesuffix('.mp3')}/vocals.mp3", "vocals_whisper_segments.json", "output.json", LANG)
    print("[pipeline] aligned successfully.")
    print("requesting")
    print("[pipeline] done.")
    #python3 '/Users/levi/Code/Sonex/test_audio/audio_sync_player.py' '/Users/levi/Code/Sonex/test_audio/RREGATON_ExperimentoMykeTowers.mp3' --json '/Users/levi/Code/Sonex/vocals_whisper_segments_aligned.json'
    os.system(f"python test_audio/gui_player.py --audio {SUBFOLDER}/{AUDIO} --align vocals_whisper_segments_aligned.json")

if __name__ == "__main__":
    SUBFOLDER = "test_audio"
    AUDIO = "RREGATON_ExperimentoMykeTowers.mp3"
    LANG = "en"
    print("[pipeline] letra_toolkit inited")
    print("[pipeline] separate...")
    separate(SUBFOLDER + "/" + AUDIO)
    input("Continue?")
    transcribe(f"{AUDIO.removesuffix('.mp3')}/vocals.mp3", "vocals_whisper_segments.json", language=LANG, _align=True)
    print("[pipeline] transcribed successfully.")
    #print("[pipeline] begin alignment...")
    align(f"{AUDIO.removesuffix('.mp3')}/vocals.mp3", "vocals_whisper_segments.json", "output.json", LANG)
    print("[pipeline] aligned successfully.")
    print("requesting")
    print("[pipeline] done.")
    #python3 '/Users/levi/Code/Sonex/test_audio/audio_sync_player.py' '/Users/levi/Code/Sonex/test_audio/RREGATON_ExperimentoMykeTowers.mp3' --json '/Users/levi/Code/Sonex/vocals_whisper_segments_aligned.json'
    os.system(f"python test_audio/gui_player.py --audio {SUBFOLDER}/{AUDIO} --align vocals_whisper_segments_aligned.json")
else: 
    main()