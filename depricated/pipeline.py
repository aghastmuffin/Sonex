import whisperx, os, argparse
from backbone.ltra.letra_toolkit import transcribe, align, separate


def _is_fresh(outputs, inputs):
    output_paths = [p for p in (outputs if isinstance(outputs, (list, tuple)) else [outputs])]
    input_paths = [p for p in (inputs if isinstance(inputs, (list, tuple)) else [inputs])]
    if not output_paths or any(not os.path.exists(p) for p in output_paths):
        return False
    newest_input = max(os.path.getmtime(p) for p in input_paths if os.path.exists(p))
    oldest_output = min(os.path.getmtime(p) for p in output_paths)
    return oldest_output >= newest_input

def main():
    parser = argparse.ArgumentParser(description="Sonex pipeline for audio transcription and alignment")
    parser.add_argument("--subfolder", default="test_audio", help="Subfolder containing audio files (default: test_audio)")
    parser.add_argument("--audio", default="RREGATON_ExperimentoMykeTowers.mp3", help="Audio file name or full path")
    parser.add_argument("--lang", default="en", help="Language code for transcription (default: en)")
    parser.add_argument("--runfree", default=False, action='store_true', help="Run without user prompts (default: False)")

    args = parser.parse_args()
    
    AUDIO_PATH = args.audio
    LANG = args.lang
    RUNFREE = args.runfree
    
    # If absolute path, use as-is; otherwise prepend subfolder
    if os.path.isabs(AUDIO_PATH):
        FULL_AUDIO_PATH = AUDIO_PATH
    else:
        FULL_AUDIO_PATH = os.path.join(args.subfolder, AUDIO_PATH)
    
    # Extract base name for vocals path
    AUDIO_BASE = os.path.basename(AUDIO_PATH).removesuffix('.mp3')
    AUDIO_DIR = os.path.dirname(FULL_AUDIO_PATH)
    VOCALS_PATH = os.path.join(AUDIO_BASE, "vocals.mp3")
    TRANSCRIPT_PATH = "vocals_whisper_segments.json"
    ALIGNED_PATH = "vocals_whisper_segments_aligned.json"
    
    print(f"[pipeline] Audio: {FULL_AUDIO_PATH}")
    print(f"[pipeline] Language: {LANG}")
    print("[pipeline] letra_toolkit inited")
    print("[pipeline] separate...")
    if _is_fresh(VOCALS_PATH, FULL_AUDIO_PATH):
        print("[pipeline] Using existing stems; skipping separation.")
    else:
        separate(FULL_AUDIO_PATH)
    if not RUNFREE:
        input("Continue?")
    if _is_fresh(TRANSCRIPT_PATH, VOCALS_PATH):
        print("[pipeline] Using cached transcript; skipping transcription.")
    else:
        transcribe(VOCALS_PATH, TRANSCRIPT_PATH, language=LANG, _align=False)
    print("[pipeline] transcribed successfully.")
    #print("[pipeline] begin alignment...")
    if _is_fresh(ALIGNED_PATH, [VOCALS_PATH, TRANSCRIPT_PATH]):
        print("[pipeline] Using cached alignment; skipping alignment.")
    else:
        align(VOCALS_PATH, TRANSCRIPT_PATH, ALIGNED_PATH, LANG)
    print("[pipeline] aligned successfully.")
    print("requesting")
    print("[pipeline] done.")
    #python3 '/Users/levi/Code/Sonex/test_audio/audio_sync_player.py' '/Users/levi/Code/Sonex/test_audio/RREGATON_ExperimentoMykeTowers.mp3' --json '/Users/levi/Code/Sonex/vocals_whisper_segments_aligned.json'
    os.system(f"python test_audio/gui_player.py --audio {FULL_AUDIO_PATH} --align {ALIGNED_PATH}")

if __name__ == "__main__":
    main()

def dep(): #old funct
    SUBFOLDER = "test_audio"
    AUDIO = "RREGATON_ExperimentoMykeTowers.mp3"
    LANG = "en"
    print("[pipeline] letra_toolkit inited")
    print("[pipeline] separate...")
    separate(SUBFOLDER + "/" + AUDIO)
    input("Continue?")
    if not os.path.exists(AUDIO.removesuffix('.mp3')):
        input("There was an issue with the program.")
    transcribe(f"{AUDIO.removesuffix('.mp3')}/vocals.mp3", "vocals_whisper_segments.json", language=LANG, _align=True)
    print("[pipeline] transcribed successfully.")
    #print("[pipeline] begin alignment...")
    align(f"{AUDIO.removesuffix('.mp3')}/vocals.mp3", "vocals_whisper_segments.json", "output.json", LANG)
    print("[pipeline] aligned successfully.")
    print("requesting")
    print("[pipeline] done.")
    #python3 '/Users/levi/Code/Sonex/test_audio/audio_sync_player.py' '/Users/levi/Code/Sonex/test_audio/RREGATON_ExperimentoMykeTowers.mp3' --json '/Users/levi/Code/Sonex/vocals_whisper_segments_aligned.json'
    os.system(f"python test_audio/gui_player.py --audio {SUBFOLDER}/{AUDIO} --align vocals_whisper_segments_aligned.json")