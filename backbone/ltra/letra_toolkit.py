import io, os, sys
import json
import shutil
import re
from pathlib import Path
from shutil import rmtree
import subprocess as sp
from typing import Dict, Literal, Tuple, Optional, IO
#FFMPEG, openai-whisper, demucs 
from faster_whisper import WhisperModel
import json 
"""Lyrics timing"""
MINIMUM = (3, 9, 0)
if sys.version_info < MINIMUM:
    sys.stderr.write(f"Python {MINIMUM[0]}.{MINIMUM[1]}.{MINIMUM[2]} or later is required.\n")
    sys.exit(1)

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


_DEMUCS_PROGRESS_RE = re.compile(r"(\d{1,3})%\|")


def copy_process_streams(process: sp.Popen, progress_cb=None):
    import threading

    def _reader(stream, std):
        if stream is None:
            return
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        while True:
            raw_buf = stream.read(2 ** 16)
            if not raw_buf:
                break
            buf = raw_buf.decode(errors="replace")
            std.write(buf)
            std.flush()
            if progress_cb is not None:
                for match in _DEMUCS_PROGRESS_RE.finditer(buf):
                    try:
                        progress_value = max(0, min(100, int(match.group(1))))
                    except ValueError:
                        continue
                    progress_cb(progress_value)

    threads = []
    for stream, std in [(process.stdout, sys.stdout), (process.stderr, sys.stderr)]:
        t = threading.Thread(target=_reader, args=(stream, std), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def separate(inp, outp=None, force: bool = False, demucs_progress_cb=None):
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
        sys.executable, "-m", "demucs.separate",
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
    copy_process_streams(p, progress_cb=demucs_progress_cb)
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

    vocals_path = out_dir_path / "vocals.mp3"
    vocals_normalized_path = out_dir_path / "vocals_normalized.mp3"
    if vocals_path.exists():
        print("Applying loudness normalization to vocals stem...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(vocals_path),
            "-af",
            "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            f"{mp3_rate}k",
            str(vocals_normalized_path),
        ]
        p = sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
        copy_process_streams(p)
        p.wait()
        if p.returncode == 0 and vocals_normalized_path.exists():
            vocals_normalized_path.replace(vocals_path)
            print(f"Saved normalized vocals stem: {vocals_path}")
        else:
            print("WARNING: ffmpeg loudnorm failed; keeping original vocals.mp3")


def _probe_audio_duration_seconds(inp) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(inp),
    ]
    try:
        out = sp.check_output(cmd, stderr=sp.DEVNULL, text=True).strip()
        duration = float(out)
        return duration if duration > 0 else None
    except Exception:
        return None


WHISPER_CHUNK_SEC = 55.0
MIN_WORD_DUR = 0.05


def _coerce_float(value, default=None):
    try:
        out = float(value)
        if out != out:  # NaN
            return default
        return out
    except (TypeError, ValueError):
        return default


def _words_have_valid_times(words) -> bool:
    if not words:
        return False
    prev_end = None
    for word in words:
        start = _coerce_float(word.get("start"))
        end = _coerce_float(word.get("end"))
        if start is None or end is None or end <= start:
            return False
        if prev_end is not None and start + 1e-3 < prev_end:
            return False
        prev_end = end
    return True


def sanitize_whisper_segments(segments):
    """Ensure segment and word timestamps are usable for MFA chunking."""
    if not segments:
        return segments

    sanitized = []
    prev_end = 0.0
    for seg in segments:
        seg = dict(seg)
        seg_start = _coerce_float(seg.get("start"), prev_end)
        seg_end = _coerce_float(seg.get("end"), (seg_start or prev_end) + MIN_WORD_DUR)
        if seg_start is None:
            seg_start = prev_end
        if seg_end is None or seg_end <= seg_start:
            seg_end = seg_start + MIN_WORD_DUR

        words = []
        raw_words = seg.get("words") or []
        if raw_words:
            cursor = max(seg_start, prev_end)
            for raw in raw_words:
                word = dict(raw)
                text = word.get("word", "")
                wstart = _coerce_float(word.get("start"), cursor)
                wend = _coerce_float(word.get("end"))
                if wstart is None:
                    wstart = cursor
                wstart = max(wstart, cursor)
                if wend is None or wend <= wstart:
                    wend = wstart + MIN_WORD_DUR
                word["word"] = text
                word["start"] = wstart
                word["end"] = wend
                words.append(word)
                cursor = wend
            seg_start = words[0]["start"]
            seg_end = words[-1]["end"]
        else:
            tokens = (seg.get("text") or "").split()
            if tokens:
                duration = max(seg_end - seg_start, MIN_WORD_DUR * len(tokens))
                step = duration / len(tokens)
                cursor = seg_start
                for token in tokens:
                    words.append({"word": token, "start": cursor, "end": cursor + step})
                    cursor += step
                seg_end = words[-1]["end"]

        seg["start"] = seg_start
        seg["end"] = seg_end
        seg["words"] = words
        sanitized.append(seg)
        prev_end = max(prev_end, seg_end)
    return sanitized


def _collect_whisper_segments(model, inp, *, transcribe_kwargs, audio_duration, progress_cb=None, task_label="Whisper"):
    """
    Transcribe long vocals in fixed windows so Whisper cannot skip whole sections
    after extended musical passages between speech segments.
    """
    duration = audio_duration or _probe_audio_duration_seconds(inp)
    base_kwargs = dict(transcribe_kwargs)
    base_kwargs.setdefault("condition_on_previous_text", False)

    def _emit_progress(end_sec, label_suffix=""):
        if progress_cb is None or duration is None or duration <= 0:
            return
        try:
            pct = max(0, min(100, int((float(end_sec) / float(duration)) * 100)))
            progress_cb(pct, f"{task_label} {pct}%{label_suffix}")
        except (TypeError, ValueError, ZeroDivisionError):
            return

    if duration is None or duration <= WHISPER_CHUNK_SEC:
        segments, _info = model.transcribe(inp, **base_kwargs)
        collected = list(segments)
        _emit_progress(duration or 0, " complete")
        return collected

    collected = []
    chunk_start = 0.0
    while chunk_start < duration - 0.01:
        chunk_end = min(duration, chunk_start + WHISPER_CHUNK_SEC)
        chunk_kwargs = dict(base_kwargs)
        chunk_kwargs["clip_timestamps"] = [chunk_start, chunk_end]
        segments, _info = model.transcribe(inp, **chunk_kwargs)
        for seg in segments:
            seg_start = float(getattr(seg, "start", 0.0) or 0.0)
            seg_end = float(getattr(seg, "end", 0.0) or 0.0)
            if seg_end <= chunk_start + 0.01:
                continue
            if seg_start >= chunk_end - 0.01:
                continue
            collected.append(seg)
        _emit_progress(chunk_end, f" ({int(chunk_start)}-{int(chunk_end)}s)")
        if chunk_end >= duration - 0.01:
            break
        chunk_start = chunk_end

    collected.sort(key=lambda seg: (float(getattr(seg, "start", 0.0) or 0.0), float(getattr(seg, "end", 0.0) or 0.0)))
    return collected


def transcribe(
    inp,
    beam_size=5,
    pat=2,
    best_of=3,
    outp: Optional[str] = None,
    language: Optional[str] = None,
    model_size="medium",
    task: Literal['transcribe', 'translate'] = "transcribe",
    _align=False,
    reuse_existing: bool = True,
    progress_cb=None,
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
    task = (task or "transcribe").strip().lower()
    if task not in {"transcribe", "translate"}:
        task = "transcribe"

    if outp and reuse_existing and language is not None and _is_fresh(outp, inp):
        audio_duration_for_cache = _probe_audio_duration_seconds(inp)
        if audio_duration_for_cache and audio_duration_for_cache > WHISPER_CHUNK_SEC:
            print(
                f"INFO|Refreshing long-audio transcript (> {int(WHISPER_CHUNK_SEC)}s) with chunked Whisper: {outp}",
                flush=True,
            )
        else:
            with open(outp, "r", encoding="utf-8") as f:
                transcript = json.load(f)
            transcript = sanitize_whisper_segments(transcript)
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            print(f"Using cached transcript: {outp}")
            if progress_cb is not None:
                progress_cb(100, f"Whisper {task} complete (cached)")
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

    print(f"Whisper task={task} in: {chosen_lang}")

    audio_duration = _probe_audio_duration_seconds(inp)

    transcribe_kwargs = dict(
        beam_size=beam_size,
        patience=pat,
        vad_filter=False,
        task=task,
        best_of=best_of,
        language=chosen_lang,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    # -------------------------
    # Transcribe WITH timestamps
    # -------------------------
    segments = _collect_whisper_segments(
        model,
        inp,
        transcribe_kwargs=transcribe_kwargs,
        audio_duration=audio_duration,
        progress_cb=progress_cb,
        task_label=f"Whisper {task}",
    )

    if audio_duration is None:
        audio_duration = (
            float(segments[-1].end)
            if segments and getattr(segments[-1], "end", None) is not None
            else None
        )

    if progress_cb is not None:
        progress_cb(100, f"Whisper {task} complete")

    print(f"Whisper {task} complete: {len(segments)} segments")

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

    transcript = sanitize_whisper_segments(transcript)

    # -------------------------
    # Write output
    # -------------------------
    if not outp:
        outp = Path(inp).stem + "_whisper_segments.json"

    out_path = Path(outp)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Whisper {task} complete!")
    print(f"  Whisper-ready transcript: {out_path}")
    if _align:
        print("[Letra Toolkit] _align=true was depricated in ver 0.1.2, please switch to calling align() separately.")
        align(inp, outp, outp.replace(".json", "_aligned.json"), chosen_lang) #TODO: Remove/direct user more aggressively
    return transcript, chosen_lang



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
    except Exception:
        print("Translation package already installed or installation failed. Still attempt translation.")
    translated = []
    for segment in segments:
        translated.append(argostranslate.translate.translate(segment, source, target))
    return translated

def flatten(target_file: str, target_pitch_semitones: int = -12, band: tuple = (100, 4000), force: bool = False, sr=None, mono=True):
    import librosa
    from scipy.signal import butter, lfilter
    import numpy as np
    if Path(target_file).name != "vocals.mp3" and not force:
        print("Translation layer: (bad operation) Skipping flattening since target file is not vocals.mp3 (use force=True to override)")
        raise ValueError("Translation layer: (bad operation) Skipping flattening since target file is not vocals.mp3 (use force=True to override)")
    y, sr = librosa.load(target_file, sr=sr, mono=mono)
    y_shifted = librosa.effects.pitch_shift(y, sr, n_steps=target_pitch_semitones)
    

    def bandpass_filter(data, sr, lowcut, highcut, order=6):
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)
    
    y_filtered = bandpass_filter(y_shifted, sr, band[0], band[1])

    # === NORMALIZE AUDIO ===
    y_filtered /= np.max(np.abs(y_filtered)) + 1e-6  # avoid clipping
    return y_filtered, sr



#def mfa_align():
    """
    Docstring for mfa_align
    Uses Montreal Forced Aligner to align text to audio
    Requires: MFA installation and pretrained models
    """
#    return

def _write_whisper_segments_list(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_whisper_segments(segments), f, indent=2, ensure_ascii=False)


def _merge_segment_words(original_words, aligned_words):
    if _words_have_valid_times(original_words):
        return original_words
    if _words_have_valid_times(aligned_words):
        merged = []
        for idx, aligned in enumerate(aligned_words):
            word = dict(aligned)
            if idx < len(original_words or []):
                speaker = (original_words[idx] or {}).get("speaker")
                if speaker:
                    word["speaker"] = speaker
            merged.append(word)
        return merged
    merged = original_words or aligned_words or []
    return sanitize_whisper_segments([{"id": 0, "text": " ", "start": 0.0, "end": MIN_WORD_DUR, "words": merged}])[0]["words"]


def _build_alignment_metadata(segments):
    transcript = []
    metadata = {}
    for seg in segments:
        seg_id = seg.get("id", len(transcript))
        transcript.append({
            "id": seg_id,
            "text": seg.get("text", ""),
            "start": seg.get("start"),
            "end": seg.get("end"),
        })
        metadata[seg_id] = {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "words": seg.get("words", []),
            "speaker": seg.get("speaker"),
        }
    return transcript, metadata


def _finalize_alignment_output(aligned, metadata):
    segments_out = []
    if isinstance(aligned, dict) and "segments" in aligned:
        source_segments = aligned["segments"]
    elif isinstance(aligned, list):
        source_segments = aligned
    else:
        source_segments = []

    for seg in source_segments:
        seg = dict(seg)
        seg_id = seg.get("id")
        meta = metadata.get(seg_id, {}) if seg_id is not None else {}
        seg["words"] = _merge_segment_words(meta.get("words") or [], seg.get("words") or [])
        if meta.get("start") is not None and seg.get("start") is None:
            seg["start"] = meta["start"]
        if meta.get("end") is not None and seg.get("end") is None:
            seg["end"] = meta["end"]
        if meta.get("speaker") and not seg.get("speaker"):
            seg["speaker"] = meta["speaker"]
        segments_out.append(seg)
    return sanitize_whisper_segments(segments_out)


def align(
    audio_path: str,
    transcript_path: str,
    output_path: Optional[str] = None,
    language: Optional[str] = None,
    reuse_existing: bool = True,
    return_char_alignments: bool = False,
    flatten_audio: bool = True,
    writeback_transcript: bool = True,
):
    """
    Align transcript segments from JSON to audio using WhisperX CTC alignment.
    Loads segments with existing start/end metadata and preserves them.
    """

    import json
    import torch
    import whisperx #XXX: fails to recongize RREGATON test Experimento is spanish lang and transcription qual suffers as result.
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_language = str(language or "en").strip().lower() or "en"

    if output_path is None:
        output_path = Path(transcript_path).with_stem(f"{Path(transcript_path).stem}.aligned").with_suffix(".json")

    if reuse_existing and _is_fresh(output_path, [audio_path, transcript_path]):
        with open(output_path, "r", encoding="utf-8") as f:
            aligned = json.load(f)
        with open(transcript_path, "r", encoding="utf-8") as f:
            source_segments = json.load(f)
        _, metadata = _build_alignment_metadata(source_segments)
        final_segments = _finalize_alignment_output(aligned, metadata)
        if writeback_transcript:
            _write_whisper_segments_list(transcript_path, final_segments)
        print(f"Using cached alignment: {output_path}")
        return {"segments": final_segments}

    # -------------------------
    # Load audio - proper preprocessing for whisperx
    # -------------------------
    import librosa
    if flatten_audio:
        try:
            audio, sr = flatten(audio_path, sr=16000, mono=True) #normalize frequencies for a more consistent alignment experience, especially on vocals. This is a lossy operation but can help with alignment quality. If you want to preserve original audio, set force=False and it will skip flattening if the file is not vocals.mp3
        except Exception as e:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    else:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Loaded audio: {len(audio)} samples at {sr}Hz")

    # -------------------------
    # Load transcript from JSON with metadata
    # -------------------------
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        raise RuntimeError("Transcript file is empty — nothing to align.")

    transcript, metadata = _build_alignment_metadata(segments)

    print(f"Aligning {len(transcript)} segments...")

    # -------------------------
    # Load alignment model
    # -------------------------
    try:
        align_model, align_model_metadata = whisperx.load_align_model(
            language_code=resolved_language,
            device=device
        )
        print(f"Loaded alignment model for language: {resolved_language}")
    except Exception as e:
        print(f"Failed to load alignment model: {e}")
        print("Falling back to original Whisper timestamps...")
        aligned = {"segments": transcript}

    else:
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
                return_char_alignments=bool(return_char_alignments),
            )
            print("CTC alignment successful")
        except Exception as e:
            print(f"CTC alignment failed: {e}")
            print("Using original Whisper timestamps instead...")
            aligned = {"segments": transcript}

    final_segments = _finalize_alignment_output(aligned, metadata)
    for seg in final_segments:
        seg_id = seg.get("id")
        if seg_id in metadata:
            orig_words = metadata[seg_id].get("words") or []
            aligned_words = []
            if isinstance(aligned, dict):
                for src in aligned.get("segments") or []:
                    if src.get("id") == seg_id:
                        aligned_words = src.get("words") or []
                        break
            if _words_have_valid_times(orig_words):
                print(f"  Segment {seg_id}: Using original Whisper word timings ({len(seg['words'])} words)")
            elif _words_have_valid_times(aligned_words):
                print(f"  Segment {seg_id}: Using CTC-aligned word timings ({len(seg['words'])} words)")
            elif seg.get("words"):
                print(f"  Segment {seg_id}: Repaired word timings ({len(seg['words'])} words)")

    aligned = {"segments": final_segments}
    if writeback_transcript:
        _write_whisper_segments_list(transcript_path, final_segments)

    # -------------------------
    # Save output
    # -------------------------
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned, f, indent=2, ensure_ascii=False)

    print(f"✓ Alignment written to {output_path}")

    return aligned


def _resolve_hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def segments_have_speaker_diarization(segments) -> bool:
    """True when transcript JSON includes speaker labels from diarization."""
    if not segments:
        return False
    for seg in segments:
        if (seg.get("speaker") or "").strip():
            return True
        for word in seg.get("words") or []:
            if (word.get("speaker") or "").strip():
                return True
    return False


def diarize(
    audio_path: str,
    transcript_path: str,
    output_path: Optional[str] = None,
    *,
    reuse_existing: bool = True,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    fill_nearest: bool = True,
    writeback_transcript: bool = True,
    progress_cb=None,
):
    """
    Assign speaker labels to an aligned Whisper transcript using WhisperX/pyannote.
    Requires a Hugging Face token in HF_TOKEN (or HUGGING_FACE_HUB_TOKEN).
    """
    import json
    import torch
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    transcript_path = Path(transcript_path)
    audio_path = str(audio_path)
    output_path = Path(output_path) if output_path else transcript_path

    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        raise RuntimeError("Transcript file is empty — nothing to diarize.")

    if reuse_existing and segments_have_speaker_diarization(segments) and _is_fresh(transcript_path, audio_path):
        print(f"Using cached diarization: {transcript_path}")
        if progress_cb is not None:
            progress_cb(100, "Diarization complete (cached)")
        return segments

    token = _resolve_hf_token()
    if not token:
        raise RuntimeError(
            "Speaker diarization requires a Hugging Face token. "
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running speaker diarization on {device}...")

    diarize_model = DiarizationPipeline(token=token, device=device)

    def _report_progress(pct):
        if progress_cb is None:
            return
        try:
            clamped = max(0, min(100, int(float(pct))))
        except (TypeError, ValueError):
            return
        progress_cb(clamped, f"Diarizing {clamped}%")

    diarize_df = diarize_model(
        audio_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        progress_callback=_report_progress,
    )
    result = assign_word_speakers(
        diarize_df,
        {"segments": segments},
        fill_nearest=bool(fill_nearest),
    )
    final_segments = sanitize_whisper_segments(result.get("segments") or segments)

    if writeback_transcript:
        _write_whisper_segments_list(transcript_path, final_segments)
    if output_path.resolve() != transcript_path.resolve():
        _write_whisper_segments_list(output_path, final_segments)

    if progress_cb is not None:
        progress_cb(100, "Diarization complete")

    print(f"✓ Diarization written to {transcript_path}")
    return final_segments


def select_best_transcript():
    """
    Docstring for select_best_transcript
    Uses perplexity scoring to select best transcript from multiple options
    """
    return

if __name__ == "__main__":
    def example_progress(index: int, total: int, text: Optional[str] = None):
        try:
            pct = ((index + 1) / total * 100) if total else 0
        except Exception:
            pct = 0
        if text:
            print(f"[progress] {index+1}/{total} ({pct:.1f}%) - {text[:60]}")
        else:
            print(f"[progress] {index+1}/{total} ({pct:.1f}%)")
        sys.stdout.flush()

    if not os.path.exists("ADV") and os.path.isdir("ADV"):
        separate("ADV.mp3")
    else:
        print("DEMUCS SKIPPED")

    # Example: caller receives progress callbacks during transcription
    transcribe("ADV/vocals.mp3", on_progress=example_progress)