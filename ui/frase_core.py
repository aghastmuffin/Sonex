"""Shared lyrics viewer logic (data loading, analysis, segment timing). UI-agnostic."""

from __future__ import annotations

import html
import json
import os
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np

pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BEAT_VISUAL_MS = 100
TRANSCRIPT_FILE_CANDIDATES = [
    "playback_segments.json",
    "mfa_vocals_phone_segments.json",
    "vocals_whisper_segments.json",
    "vocals_whisper_segments_aligned.json",
    "mfa_vocals_whisper_segments.json",
]
PHONE_LEVEL_TRANSCRIPTS = {"mfa_vocals_phone_segments.json", "playback_segments.json"}
TRANSLATED_TRANSCRIPT_FILE_CANDIDATES = [
    "translated.json",
    "argos_translated.json",
    "vocals_whisper_segments_translated.json",
    "whisper_translated.json",
]
DISCOVERY_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "model_offload",
    "htdemucs",
}

LOOP_ACQUIRE_SIM = 0.70
LOOP_KEEP_SIM = 0.50
LOOP_MAX_MISSES = 2
LOOP_MIN_PERIOD_MS = 1000
LOOP_MAX_PERIOD_MS = 9000
LOOP_ACQUIRE_COOLDOWN_MS = 120

SPEAKER_COLORS = ("#7eb8ff", "#ffb86c", "#b4f0a7", "#f5a3d8", "#c8a8ff", "#ffe08a")

OUTPUT_ROOT = ""


def default_output_root():
    app_name = "Sonex"
    if sys.platform.startswith("darwin"):
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    elif sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, app_name, "outputs")


def resolve_output_root(argv=None, output_root_arg=None):
    if output_root_arg:
        return output_root_arg

    env_root = os.environ.get("SONEX_OUTPUT_ROOT")
    if env_root:
        return env_root

    argv = list(argv) if argv is not None else list(sys.argv[1:])
    for arg in argv:
        if arg.startswith("--output-root="):
            return arg.split("=", 1)[1]
    for i, arg in enumerate(argv):
        if arg == "--output-root" and i + 1 < len(argv):
            return argv[i + 1]

    return default_output_root()


def init_output_root():
    global OUTPUT_ROOT
    OUTPUT_ROOT = os.path.abspath(resolve_output_root())


def _build_translated_segment_list(json_data):
    segments = _build_segment_list(json_data)
    if segments:
        return segments

    out = []
    for broad_chunk in json_data:
        text = (broad_chunk.get("text") or "").strip()
        if not text:
            continue

        seg_start = float(broad_chunk.get("start", 0.0))
        seg_end = float(broad_chunk.get("end", seg_start))
        out.append(
            {
                "start": seg_start,
                "end": seg_end,
                "text": text,
                "words": [
                    {
                        "start": seg_start,
                        "end": seg_end,
                        "word": text + " ",
                        "phones": [],
                        "phone_segments": [],
                    }
                ],
            }
        )
    return out


def segments_have_speaker_diarization(segments) -> bool:
    """True when loaded transcript data includes speaker labels."""
    if not segments:
        return False
    for seg in segments:
        if (seg.get("speaker") or "").strip():
            return True
        for word in seg.get("words") or []:
            if (word.get("speaker") or "").strip():
                return True
    return False


def collect_speaker_ids(segments) -> list[str]:
    speaker_ids = []
    seen = set()
    for seg in segments or []:
        candidates = [seg.get("speaker")]
        candidates.extend((w.get("speaker") for w in seg.get("words") or []))
        for speaker in candidates:
            speaker = (speaker or "").strip()
            if speaker and speaker not in seen:
                seen.add(speaker)
                speaker_ids.append(speaker)
    return speaker_ids


def speaker_color(speaker_id: str | None) -> str:
    if not speaker_id:
        return "#ffffff"
    text = str(speaker_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        idx = int(digits)
    else:
        idx = sum(ord(ch) for ch in text)
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]


def propagate_speaker_labels(source_segments, target_segments):
    if not source_segments or not target_segments:
        return target_segments
    for seg_idx, target_seg in enumerate(target_segments):
        if seg_idx >= len(source_segments):
            break
        source_seg = source_segments[seg_idx]
        if source_seg.get("speaker") and not target_seg.get("speaker"):
            target_seg["speaker"] = source_seg["speaker"]
        source_words = source_seg.get("words") or []
        target_words = target_seg.get("words") or []
        for word_idx, target_word in enumerate(target_words):
            if word_idx >= len(source_words):
                break
            speaker = source_words[word_idx].get("speaker")
            if speaker and not target_word.get("speaker"):
                target_word["speaker"] = speaker
    return target_segments


def _word_dict_for_render(word):
    item = {
        "start": float(word["start"]),
        "end": float(word["end"]),
        "word": str(word.get("word", "")) + " ",
        "phones": [
            {
                "phone": str(p.get("phone", "")),
                "start": float(p["start"]),
                "end": float(p["end"]),
            }
            for p in word.get("phones", [])
            if "start" in p and "end" in p
        ],
        "phone_segments": [
            {
                "text": str(ps.get("text", "")),
                "char_start": int(ps.get("char_start", 0)),
                "char_end": int(ps.get("char_end", 0)),
                "phone": str(ps.get("phone", "")),
                "start": float(ps["start"]),
                "end": float(ps["end"]),
            }
            for ps in word.get("phone_segments", [])
            if "start" in ps and "end" in ps
        ],
    }
    speaker = (word.get("speaker") or "").strip()
    if speaker:
        item["speaker"] = speaker
    return item


def _build_segment_list(json_data):
    out = []
    for broad_chunk in json_data:
        words = broad_chunk.get("words", [])
        if not words:
            continue

        seg_start = broad_chunk.get("start", words[0].get("start", 0.0))
        seg_end = broad_chunk.get("end", words[-1].get("end", seg_start))

        segment = {
            "start": float(seg_start),
            "end": float(seg_end),
            "text": broad_chunk.get("text", ""),
            "words": [_word_dict_for_render(w) for w in words],
        }
        speaker = (broad_chunk.get("speaker") or "").strip()
        if speaker:
            segment["speaker"] = speaker
        out.append(segment)
    return out


def _build_source_segment_list(json_data):
    out = []
    for broad_chunk in json_data:
        words = broad_chunk.get("source_words") or broad_chunk.get("words", [])
        if not words:
            continue

        seg_start = broad_chunk.get("start", words[0].get("start", 0.0))
        seg_end = broad_chunk.get("end", words[-1].get("end", seg_start))

        segment = {
            "start": float(seg_start),
            "end": float(seg_end),
            "text": broad_chunk.get("source_text", broad_chunk.get("text", "")),
            "words": [_word_dict_for_render(w) for w in words if "start" in w and "end" in w],
        }
        speaker = (broad_chunk.get("speaker") or "").strip()
        if speaker:
            segment["speaker"] = speaker
        out.append(segment)
    return out


def _has_embedded_source_fields(json_data):
    for seg in json_data:
        if seg.get("source_text"):
            return True
        src_words = seg.get("source_words")
        if isinstance(src_words, list) and len(src_words) > 0:
            return True
    return False


def _load_segments_from_file(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    return _build_segment_list(json_data)


def _load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def resolve_audio_path(transcript_file):
    parent = os.path.dirname(transcript_file)
    parent_name = os.path.basename(parent)
    audio_path = os.path.join(parent, f"{parent_name}.mp3")
    if os.path.exists(audio_path):
        return audio_path
    return None


def _resolve_analysis_frame_ms(data, num_frames):
    if num_frames <= 0:
        return 1.0

    if "frame_ms" in data:
        arr = np.asarray(data["frame_ms"], dtype=np.float64)
        if arr.size > 0 and float(arr.flat[0]) > 0:
            return float(arr.flat[0])

    if "analysis_duration_sec" in data:
        dur = np.asarray(data["analysis_duration_sec"], dtype=np.float64)
        if dur.size > 0 and float(dur.flat[0]) > 0:
            return (float(dur.flat[0]) * 1000.0) / float(num_frames)

    sample_rate = hop_size = None
    if "sample_rate" in data and len(data["sample_rate"]) > 0:
        sample_rate = float(data["sample_rate"][0])
    if "hop_size" in data and len(data["hop_size"]) > 0:
        hop_size = float(data["hop_size"][0])
    nominal_ms = (
        (hop_size * 1000.0) / sample_rate
        if sample_rate and hop_size and sample_rate > 0
        else 1.0
    )

    inferred_duration = None
    if "beat_times" in data:
        beat_times = np.asarray(data["beat_times"], dtype=np.float64)
        if beat_times.size > 0:
            inferred_duration = float(np.max(beat_times)) * 1.02
    if inferred_duration and inferred_duration > 0:
        corrected_ms = (inferred_duration * 1000.0) / float(num_frames)
        if corrected_ms > nominal_ms * 1.25:
            return corrected_ms

    return max(nominal_ms, 1e-6)


def _prefer_native_transcript_path(folder, fallback=None):
    if folder:
        native_path, _folder = _resolve_native_transcript_in_dir(folder)
        if native_path:
            return native_path
    return fallback


def _resolve_native_transcript_in_dir(path):
    first_valid = None

    for name in TRANSCRIPT_FILE_CANDIDATES:
        candidate = os.path.join(path, name)
        if not os.path.exists(candidate):
            continue
        if _looks_like_corrupt_json(candidate):
            continue

        if first_valid is None:
            first_valid = (candidate, path)

        if name in PHONE_LEVEL_TRANSCRIPTS and not _has_valid_phone_timing(candidate):
            continue

        return candidate, path

    if first_valid:
        return first_valid
    return None, None


def _resolve_translated_transcript_in_dir(path):
    for name in TRANSLATED_TRANSCRIPT_FILE_CANDIDATES:
        candidate = os.path.join(path, name)
        if not os.path.exists(candidate):
            continue
        if _looks_like_corrupt_json(candidate):
            continue
        return candidate, path
    return None, None


def _resolve_transcripts_in_dir(path):
    native_path, native_folder = _resolve_native_transcript_in_dir(path)
    translated_path, translated_folder = _resolve_translated_transcript_in_dir(path)

    if native_path or translated_path:
        return native_path, translated_path, (native_folder or translated_folder)
    return None, None, None


def _looks_like_corrupt_json(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return not isinstance(data, list)
    except Exception:
        return True


def _word_has_phone_timing(word):
    for p in word.get("phones", []) or []:
        try:
            ps = float(p.get("start"))
            pe = float(p.get("end"))
            if pe > ps:
                return True
        except (TypeError, ValueError, KeyError):
            continue
    for ps in word.get("phone_segments", []) or []:
        try:
            pstart = float(ps.get("start"))
            pend = float(ps.get("end"))
            if pend > pstart:
                return True
        except (TypeError, ValueError, KeyError):
            continue
    return False


def _has_valid_phone_timing(path, min_coverage=0.85):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return False

    if not isinstance(data, list):
        return False

    total_words = 0
    words_with_phones = 0
    for seg in data:
        for w in seg.get("words", []) or []:
            total_words += 1
            if _word_has_phone_timing(w):
                words_with_phones += 1

    if total_words <= 0:
        return False

    coverage = float(words_with_phones) / float(total_words)
    return coverage >= float(min_coverage)


def _find_transcript_files(folder_path):
    direct_native, direct_translated, direct_folder = _resolve_transcripts_in_dir(folder_path)
    if direct_native and direct_translated:
        return direct_native, direct_translated, direct_folder

    partial_match = (
        (direct_native, direct_translated, direct_folder)
        if (direct_native or direct_translated)
        else (None, None, None)
    )

    for root, _, _ in os.walk(folder_path):
        nested_native, nested_translated, nested_folder = _resolve_transcripts_in_dir(root)
        if nested_native and nested_translated:
            return nested_native, nested_translated, nested_folder
        if (nested_native or nested_translated) and not partial_match[2]:
            partial_match = (nested_native, nested_translated, nested_folder)

    return partial_match


def shorten_path(path, max_len=64):
    if not path:
        return "(not selected)"
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


def discover_eligible_generated_dirs(search_root, max_results=300):
    if not search_root or not os.path.isdir(search_root):
        return []

    found = []
    for root, dirs, _files in os.walk(search_root):
        dirs[:] = [
            d
            for d in dirs
            if d not in DISCOVERY_SKIP_DIR_NAMES
            and not d.startswith(".")
            and not d.startswith("_")
        ]

        native_path, translated_path, resolved = _resolve_transcripts_in_dir(root)
        if native_path is None and translated_path is None:
            continue

        try:
            rel = os.path.relpath(root, search_root)
        except ValueError:
            rel = root
        if rel == ".":
            rel = os.path.basename(root)

        mode = "dual" if (native_path and translated_path) else ("native" if native_path else "translated")
        found.append(
            {
                "label": f"{rel} [{mode}]",
                "folder": resolved,
                "native": native_path,
                "translated": translated_path,
            }
        )
        if len(found) >= max_results:
            break

    found.sort(key=lambda item: item["label"].lower())
    return found


def refresh_eligible_dirs():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    scan_roots = [OUTPUT_ROOT, repo_root]
    seen = set()
    merged = []
    for root in scan_roots:
        for item in discover_eligible_generated_dirs(root):
            folder = item["folder"]
            if folder in seen:
                continue
            seen.add(folder)
            merged.append(item)
    merged.sort(key=lambda item: item["label"].lower())
    return merged


def _find_analysis_file(folder_path, output_root=None):
    if not folder_path:
        return None

    base_name = os.path.basename(folder_path.rstrip("/"))
    preferred = [
        f"{base_name}_novocs_analysis.npz",
        f"{base_name}_vocs_analysis.npz",
        f"{base_name}_analysis.npz",
    ]

    search_roots = []
    if output_root:
        search_roots.append(output_root)
    search_roots.append(folder_path)

    for root in search_roots:
        for name in preferred:
            candidate = os.path.join(root, name)
            if os.path.exists(candidate):
                return candidate

        try:
            for name in os.listdir(root):
                if name.startswith(base_name) and name.endswith("_analysis.npz"):
                    return os.path.join(root, name)
        except OSError:
            continue

    return None


def _extract_fuzzy_note_runs(
    hpcp,
    top_k=3,
    half_life_ms=160,
    frame_ms=1.0,
    bridge_ms=130,
    min_run_ms=120,
    min_relative_strength=0.45,
):
    n = len(hpcp)
    if n == 0:
        return []

    frame_ms = max(1e-6, float(frame_ms))
    half_life_ms = max(1e-6, float(half_life_ms))
    decay = 0.5 ** (frame_ms / half_life_ms)
    bridge_frames = max(1, int(round(float(bridge_ms) / frame_ms)))
    min_run_frames = max(1, int(round(float(min_run_ms) / frame_ms)))

    state = np.zeros(12, dtype=np.float64)
    dominant = np.zeros(n, dtype=np.int16)

    for i in range(n):
        frame = np.asarray(hpcp[i], dtype=np.float64)
        if frame.ndim > 1:
            frame = frame.reshape(-1)
        if len(frame) > len(pitch_classes):
            frame = frame[: len(pitch_classes)]

        state *= decay

        peak = float(frame.max()) if len(frame) > 0 else 0.0
        if peak < 1e-6:
            dominant[i] = dominant[i - 1] if i > 0 else 0
            continue

        ranks = np.argsort(frame)[::-1]
        for r in range(min(top_k, len(ranks), len(pitch_classes))):
            idx = int(ranks[r])
            if float(frame[idx]) / peak >= min_relative_strength:
                state[idx] += 1.0 / (r + 1.0)

        dominant[i] = int(np.argmax(state))

    runs = []
    cur_note = int(dominant[0])
    start = 0
    for i in range(1, n):
        if int(dominant[i]) != cur_note:
            runs.append([cur_note, start, i - 1])
            cur_note = int(dominant[i])
            start = i
    runs.append([cur_note, start, n - 1])

    bridged = []
    i = 0
    while i < len(runs):
        if i + 2 < len(runs):
            a_note, a_s, a_e = runs[i]
            _b_note, b_s, b_e = runs[i + 1]
            c_note, _c_s, c_e = runs[i + 2]
            if a_note == c_note and (b_e - b_s + 1) <= bridge_frames:
                bridged.append([a_note, a_s, c_e])
                i += 3
                continue
        bridged.append(runs[i])
        i += 1

    compact = []
    for run in bridged:
        note, s, e = run
        if (e - s + 1) < min_run_frames and compact:
            compact[-1][2] = e
        else:
            compact.append([note, s, e])

    return compact


def _run_index_for_frame(runs, frame_idx):
    n = len(runs)
    if n == 0 or frame_idx < runs[0][1]:
        return None
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if runs[mid][1] <= frame_idx:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _detect_repeating_cycle(
    runs,
    cur_idx,
    frame_ms,
    min_cycle_notes=3,
    max_cycle_notes=40,
    lookback_runs=220,
    min_similarity=LOOP_ACQUIRE_SIM,
):
    frame_ms = max(1e-6, float(frame_ms))
    min_period_frames = LOOP_MIN_PERIOD_MS / frame_ms
    max_period_frames = LOOP_MAX_PERIOD_MS / frame_ms

    start_idx = max(0, cur_idx - lookback_runs + 1)
    seq = [int(r[0]) for r in runs[start_idx: cur_idx + 1]]
    rel_cur = len(seq) - 1

    best = None
    for cycle_len in range(min_cycle_notes, max_cycle_notes + 1):
        if rel_cur + 1 < cycle_len * 2:
            break

        b = seq[rel_cur - cycle_len + 1: rel_cur + 1]
        if len(set(b)) < 3:
            continue

        anchor_start = start_idx + (rel_cur - cycle_len + 1)
        period_frames = max(1, int(runs[cur_idx][2]) - int(runs[anchor_start][1]) + 1)
        if period_frames < min_period_frames or period_frames > max_period_frames:
            continue

        a1 = seq[rel_cur - (2 * cycle_len) + 1: rel_cur - cycle_len + 1]
        sim1 = SequenceMatcher(None, a1, b).ratio()
        if sim1 < min_similarity:
            continue

        if rel_cur + 1 >= cycle_len * 3:
            a2 = seq[rel_cur - (3 * cycle_len) + 1: rel_cur - (2 * cycle_len) + 1]
            sim2 = SequenceMatcher(None, a2, b).ratio()
            if sim2 < min_similarity - 0.12:
                continue
            sim = 0.5 * (sim1 + sim2)
        else:
            sim = sim1

        score = (round(sim, 2), -cycle_len)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "sim": sim,
                "cycle_len": cycle_len,
                "anchor_start": anchor_start,
                "anchor_frame": int(runs[anchor_start][1]),
                "period_frames": period_frames,
                "motif": [int(runs[k][0]) for k in range(anchor_start, cur_idx + 1)],
            }

    return best


def tokenize_for_render(seg_words):
    tokens = []
    for w in seg_words:
        s = w["word"]
        lead = 0
        while lead < len(s) and s[lead] == " ":
            lead += 1
        tokens.append((lead, s[lead:], w.get("speaker")))
    return tokens


def compute_partial_highlight(word, elapsed):
    phone_segments = word.get("phone_segments", [])
    if phone_segments:
        phone_idx = None
        for pi, p in enumerate(phone_segments):
            if p["start"] <= elapsed < p["end"]:
                phone_idx = pi
                break
        if phone_idx is None and elapsed >= phone_segments[-1]["end"]:
            phone_idx = len(phone_segments) - 1
        if phone_idx is not None and len(phone_segments) > 0:
            cur = phone_segments[phone_idx]
            return {
                "char_start": int(cur.get("char_start", 0)),
                "char_end": int(cur.get("char_end", 0)),
            }
        return None

    phones = word.get("phones", [])
    if phones:
        phone_idx = None
        for pi, p in enumerate(phones):
            if p["start"] <= elapsed < p["end"]:
                phone_idx = pi
                break
        if phone_idx is None and elapsed >= phones[-1]["end"]:
            phone_idx = len(phones) - 1
        if phone_idx is not None and len(phones) > 0:
            return {
                "start_frac": phone_idx / len(phones),
                "end_frac": (phone_idx + 1) / len(phones),
            }
    return None


def tokens_to_html(tokens, highlight_idx, partial_highlight=None, *, use_speaker_colors=False):
    parts = []
    for idx, token in enumerate(tokens):
        lead_spaces, text = token[0], token[1]
        speaker = token[2] if len(token) > 2 else None
        if lead_spaces:
            parts.append("&nbsp;" * lead_spaces)
        if not text:
            continue

        escaped = html.escape(text)
        if partial_highlight and idx == highlight_idx:
            n_chars = len(text)
            if "char_start" in partial_highlight and "char_end" in partial_highlight:
                i0 = max(0, min(n_chars, int(partial_highlight.get("char_start", 0))))
                i1 = max(i0, min(n_chars, int(partial_highlight.get("char_end", n_chars))))
            else:
                i0 = max(0, min(n_chars, int(n_chars * partial_highlight.get("start_frac", 0.0))))
                i1 = max(i0, min(n_chars, int(n_chars * partial_highlight.get("end_frac", 1.0))))
            pre = html.escape(text[:i0])
            mid = html.escape(text[i0:i1])
            post = html.escape(text[i1:])
            parts.append(f'{pre}<span style="color:#ffdc50">{mid}</span>{post}')
        elif idx == highlight_idx:
            parts.append(f'<span style="color:#ffdc50">{escaped}</span>')
        elif use_speaker_colors and speaker:
            parts.append(f'<span style="color:{speaker_color(speaker)}">{escaped}</span>')
        else:
            parts.append(escaped)
    return "".join(parts)


@dataclass
class TrackState:
    seg_i: int = 0
    word_i: int = 0
    cached_seg_id: Any = None
    cached_tokens: list | None = None


@dataclass
class LyricsSession:
    segments: list = field(default_factory=list)
    segments1: list = field(default_factory=list)
    mode_name: str = ""
    header_text: str = ""
    audio_path: str | None = None
    native_file: str | None = None
    translated_file: str | None = None
    resolved_folder: str | None = None
    has_speaker_diarization: bool = False
    speaker_ids: list = field(default_factory=list)

    analysis_hpcp: Any = None
    analysis_beats: Any = None
    analysis_bpm: float | None = None
    analysis_note_strengths: Any = None
    analysis_beat_times_ms: Any = None
    analysis_frame_ms: float = 1.0
    analysis_note_runs: list = field(default_factory=list)

    note_bar_levels: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    last_beat_found_at: int = -1000
    loop_state: dict = field(default_factory=lambda: {
        "active": False,
        "motif": [],
        "motif_text": "",
        "anchor_frame": 0,
        "period_frames": 1,
        "cycle_len": 0,
        "misses": 0,
        "last_attempt_frame": -10 ** 9,
    })

    orig: TrackState = field(default_factory=TrackState)
    trans: TrackState = field(default_factory=TrackState)

    def reset_loop_state(self):
        self.loop_state.update(
            active=False,
            motif=[],
            motif_text="",
            anchor_frame=0,
            period_frames=1,
            cycle_len=0,
            misses=0,
            last_attempt_frame=-10 ** 9,
        )

    def reset_playback_indices(self):
        self.orig = TrackState()
        self.trans = TrackState()
        self.last_beat_found_at = -1000

    def load_analysis_data(self, folder_path):
        self.analysis_hpcp = None
        self.analysis_beats = None
        self.analysis_bpm = None
        self.analysis_note_strengths = None
        self.analysis_beat_times_ms = None
        self.analysis_frame_ms = 1.0
        self.analysis_note_runs = []
        self.note_bar_levels = np.zeros(12, dtype=np.float32)
        self.reset_loop_state()

        analysis_path = _find_analysis_file(folder_path, output_root=OUTPUT_ROOT)
        if not analysis_path:
            return

        try:
            data = np.load(analysis_path)
            self.analysis_hpcp = data["hpcp"] if "hpcp" in data else None
            self.analysis_note_strengths = (
                data["note_strengths_per_frame"] if "note_strengths_per_frame" in data else None
            )
            if self.analysis_note_strengths is None:
                self.analysis_note_strengths = self.analysis_hpcp
            self.analysis_beats = data["beats"] if "beats" in data else None
            beat_times = data["beat_times"] if "beat_times" in data else None
            if beat_times is not None and len(beat_times) > 0:
                self.analysis_beat_times_ms = np.rint(
                    np.asarray(beat_times, dtype=np.float64) * 1000.0
                ).astype(np.int64)

            num_frames = len(self.analysis_hpcp) if self.analysis_hpcp is not None else (
                len(self.analysis_note_strengths) if self.analysis_note_strengths is not None else 0
            )
            self.analysis_frame_ms = _resolve_analysis_frame_ms(data, num_frames)

            bpm_arr = data["bpm"] if "bpm" in data else None
            if bpm_arr is not None and len(bpm_arr) > 0:
                self.analysis_bpm = float(bpm_arr[0])
            if self.analysis_hpcp is None and self.analysis_note_strengths is not None:
                self.analysis_hpcp = self.analysis_note_strengths
            if self.analysis_hpcp is not None and len(self.analysis_hpcp) > 0:
                self.analysis_note_runs = _extract_fuzzy_note_runs(
                    self.analysis_hpcp, frame_ms=self.analysis_frame_ms
                )
        except Exception:
            self.analysis_hpcp = None
            self.analysis_beats = None
            self.analysis_bpm = None
            self.analysis_note_strengths = None
            self.analysis_beat_times_ms = None
            self.analysis_frame_ms = 1.0
            self.analysis_note_runs = []
            self.note_bar_levels = np.zeros(12, dtype=np.float32)

    @classmethod
    def load(
        cls,
        selected_native_file,
        selected_translated_file,
        resolved_folder,
        force_native,
    ):
        session = cls()
        session.native_file = selected_native_file
        session.translated_file = selected_translated_file
        session.resolved_folder = resolved_folder

        audio_anchor_file = selected_native_file or selected_translated_file

        if force_native and selected_native_file:
            session.segments = _load_segments_from_file(selected_native_file)
            session.segments1 = []
            session.mode_name = "Native only"
        elif selected_native_file and selected_translated_file:
            translated_json = _load_json_file(selected_translated_file)
            session.segments1 = _build_translated_segment_list(translated_json)
            if session.segments1 and _has_embedded_source_fields(translated_json):
                session.segments = _build_source_segment_list(translated_json)
                session.mode_name = "Dual transcript (embedded source)"
            elif session.segments1:
                native_path = _prefer_native_transcript_path(resolved_folder, selected_native_file)
                session.segments = _load_segments_from_file(native_path)
                session.mode_name = "Dual transcript"
            else:
                native_path = _prefer_native_transcript_path(resolved_folder, selected_native_file)
                session.segments = _load_segments_from_file(native_path)
                session.segments1 = []
                session.mode_name = "Single native (translation empty)"
        elif selected_native_file:
            native_path = _prefer_native_transcript_path(resolved_folder, selected_native_file)
            session.segments = _load_segments_from_file(native_path)
            session.segments1 = []
            session.mode_name = "Single native"
        else:
            translated_json = _load_json_file(selected_translated_file)
            session.segments1 = _build_translated_segment_list(translated_json)
            if session.segments1 and _has_embedded_source_fields(translated_json):
                session.segments = _build_source_segment_list(translated_json)
                session.mode_name = "Dual transcript (embedded source)"
            elif session.segments1:
                phone_native = _prefer_native_transcript_path(resolved_folder)
                if phone_native:
                    session.segments = _load_segments_from_file(phone_native)
                    session.mode_name = "Dual transcript"
                else:
                    session.segments = _build_segment_list(translated_json)
                    session.segments1 = []
                    session.mode_name = "Single translated"
            else:
                session.segments = _build_segment_list(translated_json)
                session.segments1 = []
                session.mode_name = "Single translated"

        session._apply_diarization_metadata()
        session.header_text = (
            f"{audio_anchor_file.split('/')[-2]} | {session.mode_name}{session._diarization_header_suffix()} | taeson.co"
        )
        session.audio_path = resolve_audio_path(audio_anchor_file)
        session.load_analysis_data(resolved_folder)
        session.reset_playback_indices()
        return session

    def _diarization_header_suffix(self) -> str:
        if not self.has_speaker_diarization:
            return ""
        count = len(self.speaker_ids)
        if count > 0:
            label = "speaker" if count == 1 else "speakers"
            return f" | {count} {label}"
        return " | Diarized"

    def _apply_diarization_metadata(self):
        self.has_speaker_diarization = segments_have_speaker_diarization(self.segments)
        self.speaker_ids = collect_speaker_ids(self.segments) if self.has_speaker_diarization else []
        if self.segments1:
            if self.has_speaker_diarization:
                propagate_speaker_labels(self.segments, self.segments1)
            elif segments_have_speaker_diarization(self.segments1):
                self.has_speaker_diarization = True
                self.speaker_ids = collect_speaker_ids(self.segments1)

    def _analysis_index_from_ms(self, ms, series_len):
        if series_len <= 0:
            return -1
        if ms < 0:
            return -1
        frame_ms = max(1e-6, float(self.analysis_frame_ms))
        idx = int(round(float(ms) / frame_ms))
        return max(0, min(series_len - 1, idx))

    def segment_html_at(self, elapsed_ms, track="orig"):
        elapsed = elapsed_ms / 1000.0
        if elapsed <= 1:
            return ""

        if track == "orig":
            seg_list = self.segments
            state = self.orig
        else:
            seg_list = self.segments1
            state = self.trans

        if not seg_list:
            return ""

        si, wi = state.seg_i, state.word_i
        cache_id, cache_tokens = state.cached_seg_id, state.cached_tokens

        while si < len(seg_list) - 1:
            cur_words = seg_list[si].get("words") or []
            next_words = seg_list[si + 1].get("words") or []
            if not cur_words or not next_words:
                break
            if elapsed >= float(next_words[0]["start"]):
                si += 1
                wi = 0
                cache_id = None
                cache_tokens = None
                continue
            break

        if si >= len(seg_list):
            state.seg_i, state.word_i = si, wi
            state.cached_seg_id, state.cached_tokens = cache_id, cache_tokens
            return ""

        seg = seg_list[si]
        words = seg.get("words") or []
        if not words:
            state.seg_i, state.word_i = si, wi
            state.cached_seg_id, state.cached_tokens = cache_id, cache_tokens
            return ""

        first_start = float(words[0]["start"])
        last_end = float(words[-1]["end"])
        if elapsed < first_start or elapsed > last_end:
            state.seg_i, state.word_i = si, wi
            state.cached_seg_id, state.cached_tokens = cache_id, cache_tokens
            return ""

        while wi < len(words) and elapsed >= words[wi]["end"]:
            wi += 1

        seg_id = id(seg)
        if cache_id != seg_id:
            cache_tokens = tokenize_for_render(words)
            cache_id = seg_id

        highlight_idx = wi if wi < len(words) else -1
        partial_highlight = None
        if wi < len(words):
            partial_highlight = compute_partial_highlight(words[wi], elapsed)

        state.seg_i, state.word_i = si, wi
        state.cached_seg_id, state.cached_tokens = cache_id, cache_tokens

        if cache_tokens is None:
            return ""

        return tokens_to_html(
            cache_tokens,
            highlight_idx,
            partial_highlight,
            use_speaker_colors=self.has_speaker_diarization,
        )

    def update_note_bars(self, ms, dt_ms):
        source = self.analysis_note_strengths if self.analysis_note_strengths is not None else self.analysis_hpcp
        if source is None:
            return None

        frame_index = self._analysis_index_from_ms(ms, len(source))
        if frame_index < 0:
            return None

        frame = np.asarray(source[frame_index], dtype=np.float32)
        if frame.ndim == 0 or len(frame) < 12:
            return None

        frame = np.clip(frame[:12], 0.0, None)
        peak = float(np.max(frame))
        target = frame / peak if peak > 1e-6 else np.zeros(12, dtype=np.float32)

        if self.note_bar_levels is None or len(self.note_bar_levels) != 12:
            self.note_bar_levels = np.zeros(12, dtype=np.float32)

        dt_scale = max(0.35, min(3.0, float(dt_ms) / 16.67))
        attack = min(1.0, 0.36 * dt_scale)
        release = min(1.0, 0.14 * dt_scale)
        for i in range(12):
            delta = float(target[i]) - float(self.note_bar_levels[i])
            self.note_bar_levels[i] += delta * (attack if delta >= 0 else release)

        return self.note_bar_levels.copy()

    def _lock_loop(self, cand):
        self.loop_state.update(
            active=True,
            misses=0,
            anchor_frame=cand["anchor_frame"],
            period_frames=cand["period_frames"],
            cycle_len=cand["cycle_len"],
            motif=cand["motif"],
            motif_text=" ".join(pitch_classes[n] for n in cand["motif"]),
        )

    def update_loop_tracker(self, frame_idx):
        st = self.loop_state
        runs = self.analysis_note_runs
        if not runs or frame_idx < 0:
            self.reset_loop_state()
            return None

        cur_idx = _run_index_for_frame(runs, frame_idx)
        if cur_idx is None:
            self.reset_loop_state()
            return None

        if st["active"]:
            elapsed = frame_idx - st["anchor_frame"]
            if elapsed < 0:
                self.reset_loop_state()
            else:
                period = max(1, int(st["period_frames"]))
                if elapsed >= period:
                    cand = _detect_repeating_cycle(
                        runs, cur_idx, self.analysis_frame_ms, min_similarity=LOOP_KEEP_SIM
                    )
                    st["anchor_frame"] += (elapsed // period) * period
                    if cand is not None and cand["sim"] >= LOOP_KEEP_SIM:
                        st["misses"] = 0
                        st["period_frames"] = max(
                            1, int(round(0.7 * period + 0.3 * cand["period_frames"]))
                        )
                        new_text = " ".join(pitch_classes[n] for n in cand["motif"])
                        if SequenceMatcher(None, st["motif_text"], new_text).ratio() < 0.6:
                            st["motif"] = cand["motif"]
                            st["motif_text"] = new_text
                    else:
                        st["misses"] += 1
                        if st["misses"] > LOOP_MAX_MISSES:
                            self.reset_loop_state()

                if st["active"]:
                    period = max(1, int(st["period_frames"]))
                    phase = (frame_idx - st["anchor_frame"]) % period
                    return {
                        "motif_text": st["motif_text"],
                        "progress": phase / float(period),
                        "locked": True,
                    }

        cooldown = LOOP_ACQUIRE_COOLDOWN_MS / max(1e-6, float(self.analysis_frame_ms))
        if frame_idx - st["last_attempt_frame"] < cooldown:
            return None
        st["last_attempt_frame"] = frame_idx

        cand = _detect_repeating_cycle(
            runs, cur_idx, self.analysis_frame_ms, min_similarity=LOOP_ACQUIRE_SIM
        )
        if cand is None:
            return None

        self._lock_loop(cand)
        period = max(1, int(st["period_frames"]))
        phase = (frame_idx - st["anchor_frame"]) % period
        return {
            "motif_text": st["motif_text"],
            "progress": phase / float(period),
            "locked": True,
        }

    def loop_display_at(self, ms):
        source = self.analysis_note_strengths if self.analysis_note_strengths is not None else self.analysis_hpcp
        if source is None:
            return None
        frame_index = self._analysis_index_from_ms(ms, len(source))
        if frame_index < 0:
            return None
        return self.update_loop_tracker(frame_index)

    def beat_display_at(self, ms):
        beat_visible = False
        if self.analysis_beat_times_ms is not None and len(self.analysis_beat_times_ms) > 0:
            beat_idx = int(np.searchsorted(self.analysis_beat_times_ms, ms, side="right") - 1)
            if beat_idx >= 0:
                self.last_beat_found_at = int(self.analysis_beat_times_ms[beat_idx])
                beat_visible = (ms - self.last_beat_found_at) < BEAT_VISUAL_MS
        elif self.analysis_beats is not None:
            beat_index = self._analysis_index_from_ms(ms, len(self.analysis_beats))
            if beat_index >= 0 and self.analysis_beats[beat_index]:
                self.last_beat_found_at = ms
            beat_visible = (ms - self.last_beat_found_at) < BEAT_VISUAL_MS

        bpm_text = f"{self.analysis_bpm:5.1f}" if self.analysis_bpm is not None else "  n/a"
        beat_label = "[  BEAT  ]" if beat_visible else "          "
        return bpm_text, beat_label, beat_visible


init_output_root()
