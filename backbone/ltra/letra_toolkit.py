import io, os, select, sys, re
from pathlib import Path
from shutil import rmtree
import subprocess as sp
from typing import Dict, Tuple, Optional, IO
#FFMPEG, openai-whisper, demucs 
from faster_whisper import WhisperModel
from backbone.ltra._NLLB import translate



MINIMUM = (3, 9, 0)
if sys.version_info < MINIMUM:
    sys.stderr.write(f"Python {MINIMUM[0]}.{MINIMUM[1]}.{MINIMUM[2]} or later is required.\n")
    sys.exit(1)

detected = None #define detected in global scope for translate_advanced
#Config
model = "htdemucs"
mp3 = True
mp3_rate = 320
float32 = False  # output as float 32 wavs, unused if 'mp3' is True.
int24 = False  
two_stems = "vocals"


def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()


def separate(inp):
    inp_path = Path(inp)
    out_dir = inp_path.stem
    out_dir_path = Path(out_dir)
    if out_dir_path.exists():
        rmtree(out_dir_path)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "-m", "demucs.separate",
        "-o", str(out_dir_path),
        "-n", model
    ]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]

    files = [str(inp_path)]
    print("Going to separate the file:")
    print(str(inp_path))
    print("With command: ", " ".join(cmd + files))
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")
        return

    # Move/rename output files to vocals.mp3 and other.mp3 in out_dir
    demucs_out = out_dir_path / model / inp_path.stem
    for stem in ["vocals", "other"]:
        src = demucs_out / f"{stem}.mp3"
        dst = out_dir_path / f"{stem}.mp3"
        if src.exists():
            src.replace(dst)

def transcribe(inp, outp: Optional[str] = None, language: Optional[str] = None, model_size="medium", mfa_later=False):
    """
    Transcribe audio using faster_whisper in the detected/native language only.

    - Automatic language detection (fallback to "en" if unknown)
    - VAD filtering to remove silence
    - Word-level timestamps in the native language
    - Writes a single output file with timestamps
    """
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        print(f"Using CUDA with model: {model_size}")
    except ValueError:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"Using CPU with model: {model_size}")

    # Determine language
    chosen_lang = language
    if chosen_lang is None:
        print("Detecting language...")
        try:
            detect_segments, detect_info = model.transcribe(
                inp,
                beam_size=1,
                vad_filter=True,
                language=None
            )
            _ = list(detect_segments)  # consume generator to populate detect_info
            detected = getattr(detect_info, "language", None)
            print(f"Detected language: {detected}")
        except Exception as e:
            print(f"Language detection failed: {e}")
            detected = None
        chosen_lang = detected or "en"

    print(f"Transcribing in: {chosen_lang}")
    # Native transcription (no timestamps needed)
    try:
        segments_native, info_native = model.transcribe(
            inp,
            beam_size=5,
            vad_filter=True,
            task="transcribe",
            language=chosen_lang,
            word_timestamps=False
        )
        segments_native = list(segments_native)
        print(f"Native transcription complete: {len(segments_native)} segments")
    except Exception as e:
        print(f"Native transcription error: {e}")
        segments_native = []

    # Setup output path
    if not outp:
        outp = Path(inp).stem + "_transcribed"
    native_path = Path(outp).with_name(f"{Path(outp).stem}_native.txt")

    def write_native(fp, segs):
        """Write lowercase words separated by a single space; no timestamps."""
        for seg in segs:
            text = seg.text.strip().lower()
            fp.write(" ".join(text.split()))
            fp.write("\n")

    with open(native_path, "w", encoding="utf-8") as natf:
        if segments_native:
            print(f"Writing native transcription to: {native_path}")
            write_native(natf, segments_native)
        else:
            print("No segments to write.")
    

    print("\n✓ Transcription complete!")
    print(f"  Output: {native_path}")

    return segments_native



def translate_advanced(text, targ_lang=detected, source_lang=None):
    """
    Docstring for translate_advanced
    Analyses preexisting transcription and produces timestampped version 
    Requires: Ollama, any model of your choice that supports text generation
    """
    
    return translate(text)




def translate_simple(source, target, segments):
    """
    Docstring for translate_simple
    Translates using argostranslate built-in translation
    """
        #if hanging on build deps, run python3.9 -m pip install --prefer-binary argostranslate
    import argostranslate.package
    import argostranslate.translate
    try:
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == source and x.to_code == target, available_packages
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())
    except:
        print("Translation package already installed or installation failed. Still attempt translation.")
    translated = []
    for segment in segments:
        translated.append(argostranslate.translate.translate(segment, source, target))
    return translated


def mfa_align():
    """
    Docstring for mfa_align
    Uses Montreal Forced Aligner to align text to audio
    Requires: MFA installation and pretrained models
    """
    return


if __name__ == "__main__":
    if not os.path.exists("ADV") and os.path.isdir("ADV"):
        separate("ADV.mp3")
    else:
        print("DEMUCS SKIPPED")
    transcribe("ADV/vocals.mp3")
