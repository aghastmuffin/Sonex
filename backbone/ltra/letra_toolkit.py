import io, os, select, sys, re
import shutil
from pathlib import Path
from shutil import rmtree
import subprocess as sp
from typing import Dict, Tuple, Optional, IO
#FFMPEG, openai-whisper, demucs 
from faster_whisper import WhisperModel
from backbone.ltra._NLLB import translate
from backbone.ltra.perplex import compute_perplexity


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


def _is_fresh(outputs, inputs):
    output_paths = [Path(p) for p in (outputs if isinstance(outputs, (list, tuple)) else [outputs])]
    input_paths = [Path(p) for p in (inputs if isinstance(inputs, (list, tuple)) else [inputs])]
    if not output_paths or any(not p.exists() for p in output_paths):
        return False
    newest_input = max(p.stat().st_mtime for p in input_paths if p.exists())
    oldest_output = min(p.stat().st_mtime for p in output_paths)
    return oldest_output >= newest_input


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


def separate(inp, outp=None, force: bool = False):
    inp_path = Path(inp)
    if not outp:
        out_dir = inp_path.stem
    else:
        out_dir = Path(outp)
    out_dir_path = Path(out_dir)
    expected_outputs = [out_dir_path / "vocals.mp3"]
    if two_stems is not None:
        expected_outputs.append(out_dir_path / "other.mp3")

    if not force and _is_fresh(expected_outputs, inp_path):
        print("[separate] Using existing stems; skipping demucs.")
        return

    if out_dir_path.exists() and force:
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
    print(inp_path, inp_path.stem)
    shutil.copy2(inp_path, out_dir_path / inp_path.name)


def transcribe(
    inp,
    outp: Optional[str] = None,
    language: Optional[str] = None,
    model_size: Optional[str] = "medium",
    _align=False,
    reuse_existing: bool = True,
):
    """
    Transcribe audio using faster-whisper WITH real timestamps.
    Outputs WhisperX-compatible segments with start/end populated
    from Whisper itself (no fake timings).
    """

    import json
    from pathlib import Path
    from faster_whisper import WhisperModel

    # -------------------------
    # Reuse existing transcript if fresh
    # -------------------------
    if outp and reuse_existing and language is not None and _is_fresh(outp, inp):
        with open(outp, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        print(f"Using cached transcript: {outp}")
        return transcript, None

    # -------------------------
    # Load model
    # -------------------------
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        print(f"Using CUDA with model: {model_size}")
    except ValueError:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"Using CPU with model: {model_size}")

    # -------------------------
    # Detect language (optional)
    # -------------------------
    chosen_lang = language
    if chosen_lang is None:
        print("Detecting language...")
        detect_segments, detect_info = model.transcribe(
            inp,
            beam_size=1,
            vad_filter=False,
            language=None
        )
        _ = list(detect_segments)
        chosen_lang = getattr(detect_info, "language", None) or "en"

    print(f"Transcribing in: {chosen_lang}")

    # -------------------------
    # Transcribe WITH timestamps
    # -------------------------
    segments, _ = model.transcribe(
        inp,
        beam_size=1,
        vad_filter=False,
        task="transcribe",
        language=chosen_lang,
        word_timestamps=True
    )

    segments = list(segments)
    print(f"Native transcription complete: {len(segments)} segments")

    if not segments:
        raise RuntimeError("No transcription segments produced.")

    # -------------------------
    # Build WhisperX-compatible transcript
    # -------------------------
    transcript = []

    for i, seg in enumerate(segments):
        text = " ".join(seg.text.lower().strip().split())
        if not text:
            continue

        transcript.append({
            "id": i,
            "text": text,
            "start": float(seg.start),
            "end": float(seg.end),
            # optional but useful for debugging / downstream
            "words": [
                {
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end)
                }
                for w in (seg.words or [])
                if w.start is not None and w.end is not None
            ]
        })

    # -------------------------
    # Write output
    # -------------------------
    if not outp:
        outp = Path(inp).stem + "_whisper_segments.json"

    out_path = Path(outp)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print("\n✓ Transcription complete!")
    print(f"  WhisperX-ready transcript: {out_path}")
    if _align:
        print("[Letra Toolkit] _align=true was depricated in ver 0.1.2, please switch to calling align() separately.")
        align(inp, outp, outp.replace(".json", "_aligned.json"), chosen_lang) #TODO: Remove/direct user more aggressively
    if language == None:
        return transcript, chosen_lang
    return transcript, None #TODO: Change all to fix reformatted return type



def transcribeplain(inp, outp: Optional[str] = None, language: Optional[str] = None, model_size="medium", mfa_later=False):
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


#def mfa_align():
    """
    Docstring for mfa_align
    Uses Montreal Forced Aligner to align text to audio
    Requires: MFA installation and pretrained models
    """
#    return

def align(
    audio_path: str,
    transcript_path: str,
    output_path: Optional[str] = None,
    language: str = detected or "en",
    reuse_existing: bool = True,
):
    """
    Align transcript segments from JSON to audio using WhisperX CTC alignment.
    Loads segments with existing start/end metadata and preserves them.
    """

    import json
    import torch
    import whisperx #XXX: fails to recongize RREGATON test Experimento is spanish lang and transcription qual suffers as result.
    import torchaudio
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if output_path is None:
        output_path = Path(transcript_path).with_stem(f"{Path(transcript_path).stem}.aligned").with_suffix(".json")

    if reuse_existing and _is_fresh(output_path, [audio_path, transcript_path]):
        with open(output_path, "r", encoding="utf-8") as f:
            aligned = json.load(f)
        print(f"Using cached alignment: {output_path}")
        return aligned

    # -------------------------
    # Load audio - proper preprocessing for whisperx
    # -------------------------
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Loaded audio: {len(audio)} samples at {sr}Hz")

    # -------------------------
    # Load transcript from JSON with metadata
    # -------------------------
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        raise RuntimeError("Transcript file is empty — nothing to align.")

    # Build transcript list for alignment, preserving metadata
    transcript = []
    metadata = {}
    for seg in segments:
        seg_id = seg.get("id", len(transcript))
        transcript.append({
            "id": seg_id,
            "text": seg.get("text", ""),
            "start": seg.get("start"),
            "end": seg.get("end")
        })
        metadata[seg_id] = {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "words": seg.get("words", [])
        }

    print(f"Aligning {len(transcript)} segments...")

    # -------------------------
    # Load alignment model
    # -------------------------
    try:
        align_model, align_model_metadata = whisperx.load_align_model(
            language_code=language,
            device=device
        )
        print(f"Loaded alignment model for language: {language}")
    except Exception as e:
        print(f"Failed to load alignment model: {e}")
        print("Falling back to original Whisper timestamps...")
        # Skip CTC alignment, use original timestamps
        aligned = {"segments": transcript}
        for seg in aligned["segments"]:
            seg_id = seg.get("id")
            if seg_id in metadata and metadata[seg_id]["words"]:
                seg["words"] = metadata[seg_id]["words"]
        # Save and return early
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aligned, f, indent=2, ensure_ascii=False)
        print(f"✓ Alignment (fallback) written to {output_path}")
        return aligned

    # -------------------------
    # Run forced alignment
    # -------------------------
    try:
        aligned = whisperx.align(
            transcript,
            align_model,
            align_model_metadata,
            audio,
            device,
            return_char_alignments=False
        )
        print(f"CTC alignment successful")
    except Exception as e:
        print(f"CTC alignment failed: {e}")
        print("Using original Whisper timestamps instead...")
        aligned = {"segments": transcript}
        for seg in aligned["segments"]:
            seg_id = seg.get("id")
            if seg_id in metadata and metadata[seg_id]["words"]:
                seg["words"] = metadata[seg_id]["words"]

    # -------------------------
    # Merge original metadata with alignment results
    # -------------------------
    if isinstance(aligned, dict) and "segments" in aligned:
        for seg in aligned["segments"]:
            seg_id = seg.get("id")
            if seg_id in metadata:
                # Preserve original timing metadata
                seg["original_start"] = metadata[seg_id]["start"]
                seg["original_end"] = metadata[seg_id]["end"]
                
                # IMPORTANT: Use original Whisper word timings if available
                # CTC alignment often produces worse word boundaries than Whisper's own
                if metadata[seg_id]["words"]:
                    seg["words"] = metadata[seg_id]["words"]
                    print(f"  Segment {seg_id}: Using original Whisper word timings ({len(seg['words'])} words)")
                else:
                    # Fall back to CTC-aligned words if no original words
                    print(f"  Segment {seg_id}: Using CTC-aligned words")

    # -------------------------
    # Save output
    # -------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned, f, indent=2, ensure_ascii=False)

    print(f"✓ Alignment written to {output_path}")

    return aligned




def select_best_transcript():
    """
    Docstring for select_best_transcript
    Uses perplexity scoring to select best transcript from multiple options
    """
    return

if __name__ == "__main__":
    if not os.path.exists("ADV") and os.path.isdir("ADV"):
        separate("ADV.mp3")
    else:
        print("DEMUCS SKIPPED")
    transcribe("ADV/vocals.mp3")