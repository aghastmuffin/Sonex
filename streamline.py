import os, glob
from rich.console import Console
from rich.table import Table
from rich.prompt import IntPrompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

print("initializing the lyric analysis library")
print("it's best to close all other open programs on the system before entering the ISO code.")
language_dict = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "he": "Hebrew",
    "el": "Greek",
    "th": "Thai",
    "id": "Indonesian",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian"
}
console = Console()

def choose_file(directory=".", pattern="*"):
    console.print("[bold]Scanning files...[/bold]")

    raw_files = glob.glob(os.path.join(directory, pattern))
    files = []

    # Progress while filtering (nice if there are lots of files)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,  # clears progress bar after done
    ) as progress:
        task = progress.add_task("Processing", total=len(raw_files))
        for f in raw_files:
            if os.path.isfile(f):
                files.append(f)
            progress.advance(task)

    if not files:
        console.print("[red]No files found.[/red]")
        return None

    # Display table
    table = Table(title="Select a file", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Path", style="dim")

    for i, f in enumerate(files, 1):
        table.add_row(str(i), os.path.basename(f), f)

    console.print(table)

    # Prompt until valid choice
    while True:
        try:
            choice = IntPrompt.ask("Enter number", default=1)
            if 1 <= choice <= len(files):
                return files[choice - 1]
            console.print("[red]Invalid number.[/red]")
        except Exception:
            console.print("[red]Please enter a valid number.[/red]")

SUBFOLDER = input("parent folder: ") or "test_audio"
AUDIO = choose_file(SUBFOLDER, "*")
LANG = input("ISO 639-1 Lang Code: ") or "es"

print("Accepted values successfully, beginning to load toolkit")

from backbone.ltra.letra_toolkit import transcribe, align, separate

print("splitting")
separate(AUDIO)
print("transcribing")
AUDIO_BASE = os.path.basename(AUDIO).removesuffix(".mp3")
os.makedirs(AUDIO_BASE, exist_ok=True)
_, detectlang = transcribe(f"{AUDIO_BASE}/vocals.mp3", 5, 2,f"{AUDIO_BASE}/vocals_whisper_segments.json", language=LANG, _align=True)
if detectlang:
    LANG = detectlang
print(f"lyric analysis library completed required job: {AUDIO}")
#os.system(f"python test_audio/gui_player.py --audio {SUBFOLDER}/{AUDIO} --align {OUT}/vocals_whisper_segments_aligned.json")
print("requesting advanced alignment from MFA")
try:
    from backbone.ltra import _mfa_aligner
    _mfa_aligner.generate_aligned_v2(AUDIO_BASE, acoustic=f"{language_dict[detectlang]}", dictionary=f"{language_dict[detectlang]}_mfa", allow_fuzzy=True, fuzzy_max_lookahead=8)
except Exception as e:
    print("MFA Error:", e)
    pass
print("Trying ArgosWrapper")
from backbone.ltra.argos_tranlsate import translate_file
out = translate_file(f"{AUDIO_BASE}/vocals_whisper_segments.json", from_lang=detectlang, to_lang="en", verbose=True)