import os, glob
from rich.console import Console
from rich.table import Table
from rich.prompt import IntPrompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

print("initializing the lyric analysis library")
print("it's best to close all other open programs on the system before entering the ISO code.")
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
transcribe(f"{AUDIO_BASE}/vocals.mp3", f"{AUDIO_BASE}/vocals_whisper_segments.json", language=LANG, _align=True)
print(f"lyric analysis library completed required job: {AUDIO}")
#os.system(f"python test_audio/gui_player.py --audio {SUBFOLDER}/{AUDIO} --align {OUT}/vocals_whisper_segments_aligned.json")
print("audiolib")
from backbone.rmo.AMT import *
