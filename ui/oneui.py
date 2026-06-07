import sys
import os
import json
import contextlib
from difflib import SequenceMatcher

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QThread
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout

# MUST be set before pygame import
os.environ["SDL_VIDEODRIVER"] = "dummy"

with contextlib.redirect_stdout(None):
    import pygame
    from pygame import mixer

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
WIDTH, HEIGHT = DISPLAY_WIDTH, DISPLAY_HEIGHT

# Renderer globals — initialized lazily by init_renderer()
screen = None
clock = None
font = None
dbgfont = None
start_ticks = 0
running = True
_renderer_ready = False


def surface_to_qimage(surface) -> QImage:
    w, h = surface.get_size()
    image_str = pygame.image.tostring(surface, "RGB")
    return QImage(image_str, w, h, QImage.Format.Format_RGB888).copy()


def init_renderer():
    global screen, clock, font, dbgfont, start_ticks, running, _renderer_ready
    if _renderer_ready:
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("🗣️ SONEX Lyrics, a TAESON.CO project.")

    brand_dir = os.path.join(os.path.dirname(__file__), "assets")
    icon_candidates = [
        "resolution-logo.png",
        "sonex-high-resolution-logo.png",
        "sonex-high-resolution-logo-transparent.png",
        "sonex-high-resolution-logo-grayscale.png",
        "sonex-high-resolution-logo-grayscale-transparent.png",
    ]
    for icon_name in icon_candidates:
        icon_path = os.path.join(brand_dir, icon_name)
        if os.path.exists(icon_path):
            try:
                pygame.display.set_icon(pygame.image.load(icon_path))
                break
            except pygame.error:
                pass

    clock = pygame.time.Clock()
    font_path = os.path.join(os.path.dirname(__file__), "assets", "Darker Grotesque.ttf")
    try:
        dbgfont = pygame.font.Font(font_path, 18)
        font = pygame.font.Font(font_path, 48)
    except OSError:
        dbgfont = pygame.font.SysFont(None, 18)
        font = pygame.font.SysFont(None, 48)
    start_ticks = pygame.time.get_ticks()
    running = True
    _renderer_ready = True


pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_THRESHOLD = 0.3
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

analysis_hpcp = None
analysis_beats = None
analysis_bpm = None
analysis_note_strengths = None
analysis_beat_times_ms = None
analysis_frame_ms = 1.0
last_beat_found_at = -1000
analysis_note_runs = []
note_bar_levels = np.zeros(12, dtype=np.float32)

# --- Melodic loop tracker state ---
# Once a repeating phrase is locked we keep tracking it across frames instead of
# re-detecting from scratch every frame. This keeps the circular progress smooth
# and gives the loop a tolerance for transient transcription glitches.
LOOP_ACQUIRE_SIM = 0.70   # similarity required to *start* tracking a loop
LOOP_KEEP_SIM = 0.50      # weaker similarity still accepted to *stay* locked
LOOP_MAX_MISSES = 2       # bad repetitions tolerated before dropping the lock
LOOP_MIN_PERIOD_MS = 1000     # a melodic loop is at least ~1s long
LOOP_MAX_PERIOD_MS = 9000     # ...and at most ~9s (a phrase, not a whole section)
LOOP_ACQUIRE_COOLDOWN_MS = 120  # throttle detection while no loop is locked
loop_state = {
    "active": False,
    "motif": [],           # list of pitch-class indices in one repetition
    "motif_text": "",
    "anchor_frame": 0,     # frame index where the current repetition started
    "period_frames": 1,    # duration of one repetition, in analysis frames
    "cycle_len": 0,        # number of runs in one repetition
    "misses": 0,           # consecutive failed re-validations
    "last_attempt_frame": -10 ** 9,  # last frame we ran acquisition detection
}


def _reset_loop_state():
    loop_state.update(
        active=False,
        motif=[],
        motif_text="",
        anchor_frame=0,
        period_frames=1,
        cycle_len=0,
        misses=0,
        last_attempt_frame=-10 ** 9,
    )

# --- NEW: store broad segments instead of only flat words ---
segments = []     # original segments: [{start,end,text,words:[{start,end,word}]}]
segments1 = []    # secondary transcript (translated/system language)

seg_i = 0
word_i = 0
seg_i1 = 0
word_i1 = 0

# cache so we only recompute tokens when the segment changes
_cached_seg_id = None
_cached_tokens = None
_cached_seg_id_1 = None
_cached_tokens_1 = None


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


OUTPUT_ROOT = os.path.abspath(resolve_output_root())


def _build_translated_segment_list(json_data):
    """Build translated segments, falling back to segment-level text when word timings are missing."""
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


def _build_segment_list(json_data):
    out = []
    for broad_chunk in json_data:
        # segment start/end: use chunk start/end if present, else derive from words
        words = broad_chunk.get("words", [])
        if not words:
            continue

        seg_start = broad_chunk.get("start", words[0].get("start", 0.0))
        seg_end = broad_chunk.get("end", words[-1].get("end", seg_start))

        out.append(
            {
                "start": float(seg_start),
                "end": float(seg_end),
                "text": broad_chunk.get("text", ""),
                "words": [
                    {
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "word": str(w.get("word", "")) + " ",
                        "phones": [
                            {
                                "phone": str(p.get("phone", "")),
                                "start": float(p["start"]),
                                "end": float(p["end"]),
                            }
                            for p in w.get("phones", [])
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
                            for ps in w.get("phone_segments", [])
                            if "start" in ps and "end" in ps
                        ],
                    }
                    for w in words
                ],
            }
        )
    return out


def _build_source_segment_list(json_data):
    """Build source-language segments from embedded source fields when present."""
    out = []
    for broad_chunk in json_data:
        words = broad_chunk.get("source_words") or broad_chunk.get("words", [])
        if not words:
            continue

        seg_start = broad_chunk.get("start", words[0].get("start", 0.0))
        seg_end = broad_chunk.get("end", words[-1].get("end", seg_start))

        out.append(
            {
                "start": float(seg_start),
                "end": float(seg_end),
                "text": broad_chunk.get("source_text", broad_chunk.get("text", "")),
                "words": [
                    {
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "word": str(w.get("word", "")) + " ",
                        "phones": [
                            {
                                "phone": str(p.get("phone", "")),
                                "start": float(p["start"]),
                                "end": float(p["end"]),
                            }
                            for p in w.get("phones", [])
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
                            for ps in w.get("phone_segments", [])
                            if "start" in ps and "end" in ps
                        ],
                    }
                    for w in words
                    if "start" in w and "end" in w
                ],
            }
        )
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


def _start_audio_from_transcript_file(file_path):
    global start_ticks

    parent = os.path.dirname(file_path)
    parent_name = os.path.basename(parent)
    audio_path = os.path.join(parent, f"{parent_name}.mp3")
    if not os.path.exists(audio_path):
        return

    try:
        if not mixer.get_init():
            mixer.init()
        mixer.music.stop()
        mixer.music.load(audio_path)
        mixer.music.play()
        start_ticks = pygame.time.get_ticks()
    except pygame.error:
        pass


def _resolve_analysis_frame_ms(data, num_frames):
    """Map playback ms -> analysis frame index using true per-frame spacing."""
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

    # Legacy npz from madmom: hop-based spacing is ~10x too fast vs real frame count.
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


def _analysis_index_from_ms(ms, series_len):
    if series_len <= 0:
        return -1
    if ms < 0:
        return -1
    frame_ms = max(1e-6, float(analysis_frame_ms))
    idx = int(round(float(ms) / frame_ms))
    return max(0, min(series_len - 1, idx))


def _get_elapsed_ms():
    try:
        pos = mixer.music.get_pos()
    except pygame.error:
        pos = -1
    if pos is not None and pos >= 0:
        return int(pos)
    return max(0, pygame.time.get_ticks() - start_ticks)


def _pick_folder(dialog_title):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=dialog_title, mustexist=True)
        root.destroy()
        return selected or ""
    except Exception:
        return ""


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


def _shorten_path(path, max_len=64):
    if not path:
        return "(not selected)"
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


def _discover_eligible_generated_dirs(search_root, max_results=300):
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


def _load_analysis_data(folder_path):
    global analysis_hpcp, analysis_beats, analysis_bpm, analysis_note_strengths, analysis_beat_times_ms
    global analysis_frame_ms, analysis_note_runs, note_bar_levels

    analysis_hpcp = None
    analysis_beats = None
    analysis_bpm = None
    analysis_note_strengths = None
    analysis_beat_times_ms = None
    analysis_frame_ms = 1.0
    analysis_note_runs = []
    note_bar_levels = np.zeros(12, dtype=np.float32)
    _reset_loop_state()

    analysis_path = _find_analysis_file(folder_path, output_root=OUTPUT_ROOT)
    if not analysis_path:
        return

    try:
        data = np.load(analysis_path)
        analysis_hpcp = data["hpcp"] if "hpcp" in data else None
        analysis_note_strengths = data["note_strengths_per_frame"] if "note_strengths_per_frame" in data else None
        if analysis_note_strengths is None:
            analysis_note_strengths = analysis_hpcp
        analysis_beats = data["beats"] if "beats" in data else None
        beat_times = data["beat_times"] if "beat_times" in data else None
        if beat_times is not None and len(beat_times) > 0:
            analysis_beat_times_ms = np.rint(np.asarray(beat_times, dtype=np.float64) * 1000.0).astype(np.int64)

        num_frames = len(analysis_hpcp) if analysis_hpcp is not None else (
            len(analysis_note_strengths) if analysis_note_strengths is not None else 0
        )
        analysis_frame_ms = _resolve_analysis_frame_ms(data, num_frames)

        bpm_arr = data["bpm"] if "bpm" in data else None
        if bpm_arr is not None and len(bpm_arr) > 0:
            analysis_bpm = float(bpm_arr[0])
        if analysis_hpcp is None and analysis_note_strengths is not None:
            analysis_hpcp = analysis_note_strengths
        if analysis_hpcp is not None and len(analysis_hpcp) > 0:
            analysis_note_runs = _extract_fuzzy_note_runs(
                analysis_hpcp, frame_ms=analysis_frame_ms
            )
    except Exception:
        analysis_hpcp = None
        analysis_beats = None
        analysis_bpm = None
        analysis_note_strengths = None
        analysis_beat_times_ms = None
        analysis_frame_ms = 1.0
        analysis_note_runs = []
        note_bar_levels = np.zeros(12, dtype=np.float32)


def _draw_note_strength_bars(ms, dt_ms, x=50, y=420, width=700, height=120):
    global note_bar_levels

    source = analysis_note_strengths if analysis_note_strengths is not None else analysis_hpcp
    if source is None:
        label = dbgfont.render("Note strength: [n/a]", True, (190, 190, 190))
        screen.blit(label, (x, y + height - 20))
        return

    frame_index = _analysis_index_from_ms(ms, len(source))
    if frame_index < 0:
        label = dbgfont.render("Note strength: [n/a]", True, (190, 190, 190))
        screen.blit(label, (x, y + height - 20))
        return

    frame = np.asarray(source[frame_index], dtype=np.float32)
    if frame.ndim == 0 or len(frame) < 12:
        label = dbgfont.render("Note strength: [n/a]", True, (190, 190, 190))
        screen.blit(label, (x, y + height - 20))
        return

    frame = np.clip(frame[:12], 0.0, None)
    peak = float(np.max(frame))
    if peak > 1e-6:
        target = frame / peak
    else:
        target = np.zeros(12, dtype=np.float32)

    if note_bar_levels is None or len(note_bar_levels) != 12:
        note_bar_levels = np.zeros(12, dtype=np.float32)

    dt_scale = max(0.35, min(3.0, float(dt_ms) / 16.67))
    attack = min(1.0, 0.36 * dt_scale)
    release = min(1.0, 0.14 * dt_scale)
    for i in range(12):
        delta = float(target[i]) - float(note_bar_levels[i])
        note_bar_levels[i] += delta * (attack if delta >= 0 else release)

    panel = pygame.Rect(x - 8, y - 8, width + 16, height + 16)
    pygame.draw.rect(screen, (40, 40, 40), panel, border_radius=8)
    pygame.draw.rect(screen, (62, 62, 62), panel, width=1, border_radius=8)

    gap = 6
    bar_w = max(8, int((width - gap * 11) / 12))
    bars_total_w = bar_w * 12 + gap * 11
    start_x = x + max(0, (width - bars_total_w) // 2)
    max_h = height - 24
    base_y = y + max_h

    for i, note in enumerate(pitch_classes):
        bx = start_x + i * (bar_w + gap)

        bg_rect = pygame.Rect(bx, y, bar_w, max_h)
        pygame.draw.rect(screen, (55, 55, 55), bg_rect, border_radius=4)

        level = float(max(0.0, min(1.0, note_bar_levels[i])))
        h = max(1, int(level * max_h))
        fy = base_y - h

        color = (
            int(90 + 120 * level),
            int(120 + 100 * level),
            int(190 + 55 * level),
        )
        fg_rect = pygame.Rect(bx, fy, bar_w, h)
        pygame.draw.rect(screen, color, fg_rect, border_radius=4)

        note_surf = dbgfont.render(note, True, (220, 220, 220))
        screen.blit(note_surf, (bx + (bar_w - note_surf.get_width()) // 2, y + max_h + 2))


def _extract_fuzzy_note_runs(
    hpcp,
    top_k=3,
    half_life_ms=160,
    frame_ms=1.0,
    bridge_ms=130,
    min_run_ms=120,
    min_relative_strength=0.45,
):
    """
    Build stable note runs from noisy HPCP frames.

    Key behaviour:
    - decay is derived from half_life_ms so it works correctly
    regardless of the hop size used during analysis.
    - Weights are 1/(rank+1) so a note consistently in 2nd place
    can beat one that only sporadically takes 1st.
    - min_relative_strength gates out weak notes so noise/percussion
    does not pollute the vote.
    - bridge and min_run are in ms, converted to frames internally.

    The defaults are tuned for *melodic* granularity (notes typically change
    every ~150-400ms). A short half-life keeps note transitions crisp while the
    rank-weighted decay vote still rejects single-frame transcription flukes
    (e.g. a stray 1ms note can never outvote the accumulated current note).
    """
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


def _chord_sets_similar(a_sets, b_sets, min_overlap=0.7):
    if len(a_sets) != len(b_sets):
        return 0.0
    scores = []
    for a, b in zip(a_sets, b_sets):
        if not a and not b:
            scores.append(1.0)
        elif not a or not b:
            scores.append(0.0)
        else:
            scores.append(len(a & b) / len(a | b))
    return float(np.mean(scores)) if scores else 0.0


def _run_index_for_frame(runs, frame_idx):
    """Index of the run whose [start, end] frame span contains frame_idx.

    Runs are contiguous and sorted by start frame, so we binary-search the
    start values. Returns None only when frame_idx is before the first run.
    """
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
    min_cycle_notes=3,
    max_cycle_notes=40,
    lookback_runs=220,
    min_similarity=LOOP_ACQUIRE_SIM,
):
    """Find the most plausible repeating note-order cycle ending at cur_idx.

    For each candidate ``cycle_len`` we compare the most recent window against
    the one or two windows that precede it. Requiring the phrase to match across
    *two* prior repetitions (when enough history exists) is what separates a real
    riff from a coincidental one-off match -- it is the persistence test that the
    old single-comparison logic lacked.

    The period is measured in real analysis frames (robust to repetitions that
    contain a different *number* of runs because of transcription noise) and is
    constrained to a musically plausible length window.
    """
    frame_ms = max(1e-6, float(analysis_frame_ms))
    min_period_frames = LOOP_MIN_PERIOD_MS / frame_ms
    max_period_frames = LOOP_MAX_PERIOD_MS / frame_ms

    start_idx = max(0, cur_idx - lookback_runs + 1)
    seq = [int(r[0]) for r in runs[start_idx: cur_idx + 1]]
    rel_cur = len(seq) - 1

    best = None
    for cycle_len in range(min_cycle_notes, max_cycle_notes + 1):
        if rel_cur + 1 < cycle_len * 2:
            break  # not enough history for even two repetitions of this length

        b = seq[rel_cur - cycle_len + 1: rel_cur + 1]
        if len(set(b)) < 3:
            continue  # need at least a little melodic movement to call it a loop

        anchor_start = start_idx + (rel_cur - cycle_len + 1)
        period_frames = max(1, int(runs[cur_idx][2]) - int(runs[anchor_start][1]) + 1)
        if period_frames < min_period_frames or period_frames > max_period_frames:
            continue

        a1 = seq[rel_cur - (2 * cycle_len) + 1: rel_cur - cycle_len + 1]
        sim1 = SequenceMatcher(None, a1, b).ratio()
        if sim1 < min_similarity:
            continue

        # Persistence test: when a third window is available the phrase must also
        # resemble the repetition before last, otherwise it is likely a fluke.
        if rel_cur + 1 >= cycle_len * 3:
            a2 = seq[rel_cur - (3 * cycle_len) + 1: rel_cur - (2 * cycle_len) + 1]
            sim2 = SequenceMatcher(None, a2, b).ratio()
            if sim2 < min_similarity - 0.12:
                continue
            sim = 0.5 * (sim1 + sim2)
        else:
            sim = sim1

        # Prefer strong matches; on near-ties prefer the *shorter* (fundamental)
        # period so we lock the riff rather than an integer multiple of it.
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


def _lock_loop(cand):
    loop_state.update(
        active=True,
        misses=0,
        anchor_frame=cand["anchor_frame"],
        period_frames=cand["period_frames"],
        cycle_len=cand["cycle_len"],
        motif=cand["motif"],
        motif_text=" ".join(pitch_classes[n] for n in cand["motif"]),
    )


def _update_loop_tracker(frame_idx, runs):
    """Stateful loop tracking.

    Acquires a loop with a strong similarity threshold, then keeps it locked and
    drives progress from elapsed frames (so the circle is smooth and monotonic).
    Re-validates once per repetition with a weaker threshold and tolerates a few
    bad cycles before letting go -- this is the margin for error against single
    spurious notes from librosa.
    """
    st = loop_state
    if not runs or frame_idx < 0:
        _reset_loop_state()
        return None

    cur_idx = _run_index_for_frame(runs, frame_idx)
    if cur_idx is None:
        _reset_loop_state()
        return None

    if st["active"]:
        elapsed = frame_idx - st["anchor_frame"]
        if elapsed < 0:
            # Playhead jumped backwards (seek/restart) -> drop and re-acquire.
            _reset_loop_state()
        else:
            period = max(1, int(st["period_frames"]))
            if elapsed >= period:
                # A repetition boundary passed: re-validate with tolerance.
                cand = _detect_repeating_cycle(
                    runs, cur_idx, min_similarity=LOOP_KEEP_SIM
                )
                # Always keep the ring phase-continuous by advancing the anchor
                # whole periods at a time -- never snap it, or the circle jumps.
                st["anchor_frame"] += (elapsed // period) * period
                if cand is not None and cand["sim"] >= LOOP_KEEP_SIM:
                    st["misses"] = 0
                    # Gently track tempo drift instead of snapping the period.
                    st["period_frames"] = max(
                        1, int(round(0.7 * period + 0.3 * cand["period_frames"]))
                    )
                    # Only refresh the displayed motif when it really changed,
                    # so near-identical repetitions don't make the label flicker.
                    new_text = " ".join(pitch_classes[n] for n in cand["motif"])
                    if SequenceMatcher(None, st["motif_text"], new_text).ratio() < 0.6:
                        st["motif"] = cand["motif"]
                        st["motif_text"] = new_text
                else:
                    st["misses"] += 1
                    if st["misses"] > LOOP_MAX_MISSES:
                        _reset_loop_state()

            if st["active"]:
                period = max(1, int(st["period_frames"]))
                phase = (frame_idx - st["anchor_frame"]) % period
                return {
                    "motif_text": st["motif_text"],
                    "progress": phase / float(period),
                    "locked": True,
                }

    # Throttle acquisition: detection is the expensive path, and there is no
    # point re-scanning every render frame while nothing is looping.
    cooldown = LOOP_ACQUIRE_COOLDOWN_MS / max(1e-6, float(analysis_frame_ms))
    if frame_idx - st["last_attempt_frame"] < cooldown:
        return None
    st["last_attempt_frame"] = frame_idx

    cand = _detect_repeating_cycle(runs, cur_idx, min_similarity=LOOP_ACQUIRE_SIM)
    if cand is None:
        return None

    _lock_loop(cand)
    period = max(1, int(st["period_frames"]))
    phase = (frame_idx - st["anchor_frame"]) % period
    return {
        "motif_text": st["motif_text"],
        "progress": phase / float(period),
        "locked": True,
    }


def _draw_pattern_loader(progress, center_x, center_y, radius=11, thickness=4):
    """
    Draw a hollow circular loader that fills clockwise from a 0..1 progress value.
    """
    if progress is None:
        return
    progress = max(0.0, min(1.0, float(progress)))

    # Base ring
    pygame.draw.circle(screen, (90, 90, 90), (center_x, center_y), radius, thickness)

    # Progress arc from top, clockwise
    start_angle = -np.pi / 2
    end_angle = start_angle + (2 * np.pi * progress)
    rect = pygame.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
    pygame.draw.arc(screen, (120, 255, 170), rect, start_angle, end_angle, thickness)


def choose_generated_folder(*, stop_check=None, present_frame=None):
    if stop_check is None:
        stop_check = lambda: False
    if present_frame is None:
        present_frame = pygame.display.flip
    folder_path = None
    native_file = None
    translated_file = None
    resolved_folder = None
    force_native = False

    repo_root = os.path.dirname(os.path.dirname(__file__))
    scan_roots = [OUTPUT_ROOT, repo_root]

    def _refresh_eligible_dirs():
        seen = set()
        merged = []
        for root in scan_roots:
            for item in _discover_eligible_generated_dirs(root):
                folder = item["folder"]
                if folder in seen:
                    continue
                seen.add(folder)
                merged.append(item)
        merged.sort(key=lambda item: item["label"].lower())
        return merged

    eligible_dirs = _refresh_eligible_dirs()

    dropdown_open = False
    dropdown_selected = -1
    dropdown_scroll = 0
    dropdown_item_h = 30
    dropdown_visible = 6

    dropdown_rect = pygame.Rect(70, 190, 520, 50)
    refresh_btn = pygame.Rect(610, 190, 120, 50)
    btn_folder = pygame.Rect(70, 260, 260, 52)
    force_native_btn = pygame.Rect(350, 260, 220, 52)
    start_btn = pygame.Rect(300, 500, 200, 56)

    def _apply_dropdown_selection(index):
        nonlocal folder_path, native_file, translated_file, resolved_folder, dropdown_selected
        if not (0 <= index < len(eligible_dirs)):
            return
        dropdown_selected = index
        selection = eligible_dirs[index]
        folder_path = selection["folder"]
        native_file = selection.get("native")
        translated_file = selection.get("translated")
        resolved_folder = selection["folder"]

    if eligible_dirs:
        _apply_dropdown_selection(0)

    choosing = True
    while choosing:
        if stop_check():
            return None, None, None, False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None, False

            if event.type == pygame.MOUSEWHEEL and dropdown_open and eligible_dirs:
                max_scroll = max(0, len(eligible_dirs) - dropdown_visible)
                dropdown_scroll = max(0, min(max_scroll, dropdown_scroll - event.y))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                list_x = dropdown_rect.x
                list_y = dropdown_rect.bottom + 6
                list_h = min(dropdown_visible, len(eligible_dirs)) * dropdown_item_h + 8
                list_rect = pygame.Rect(list_x, list_y, dropdown_rect.width, list_h)

                if dropdown_open:
                    # While the list overlay is up it owns the clicks: only the
                    # header (toggle) and the list items are live. Anything else
                    # just closes the dropdown so it can't trigger the buttons
                    # hidden behind the overlay.
                    if dropdown_rect.collidepoint(event.pos):
                        dropdown_open = False
                    elif eligible_dirs and list_rect.collidepoint(event.pos):
                        rel_y = event.pos[1] - (list_y + 4)
                        clicked = dropdown_scroll + (rel_y // dropdown_item_h)
                        if 0 <= clicked < len(eligible_dirs):
                            _apply_dropdown_selection(clicked)
                        dropdown_open = False
                    else:
                        dropdown_open = False
                elif dropdown_rect.collidepoint(event.pos):
                    dropdown_open = bool(eligible_dirs)
                elif refresh_btn.collidepoint(event.pos):
                    eligible_dirs = _refresh_eligible_dirs()
                    dropdown_open = False
                    dropdown_scroll = 0
                    if eligible_dirs:
                        _apply_dropdown_selection(0)
                    else:
                        dropdown_selected = -1
                        native_file = None
                        translated_file = None
                elif btn_folder.collidepoint(event.pos):
                    selected = _pick_folder("Choose generated folder (or parent folder)")
                    if selected:
                        folder_path = selected
                        native_file, translated_file, resolved_folder = _find_transcript_files(folder_path)
                elif force_native_btn.collidepoint(event.pos):
                    force_native = not force_native
                elif start_btn.collidepoint(event.pos):
                    can_start = (native_file is not None) if force_native else ((native_file is not None) or (translated_file is not None))
                    if can_start:
                        choosing = False

        can_start = (native_file is not None) if force_native else ((native_file is not None) or (translated_file is not None))

        screen.fill((24, 24, 24))
        title = font.render("Select generated folder", True, (245, 245, 245))
        screen.blit(title, ((WIDTH - title.get_width()) // 2, 70))

        pygame.draw.rect(screen, (52, 52, 52), dropdown_rect, border_radius=8)
        pygame.draw.rect(screen, (90, 90, 90), dropdown_rect, width=1, border_radius=8)
        pygame.draw.rect(screen, (58, 58, 58), refresh_btn, border_radius=8)
        pygame.draw.rect(screen, (58, 58, 58), btn_folder, border_radius=10)
        pygame.draw.rect(screen, (90, 120, 100) if force_native else (58, 58, 58), force_native_btn, border_radius=10)
        pygame.draw.rect(
            screen,
            (70, 130, 95) if can_start else (55, 55, 55),
            start_btn,
            border_radius=10,
        )

        auto_label = dbgfont.render("Auto-scan eligible folders:", True, (220, 220, 220))
        screen.blit(auto_label, (70, 164))

        selected_label = "No eligible folders found"
        if 0 <= dropdown_selected < len(eligible_dirs):
            selected_label = eligible_dirs[dropdown_selected]["label"]
        selected_txt = dbgfont.render(_shorten_path(selected_label, max_len=78), True, (245, 245, 245))
        screen.blit(selected_txt, (84, dropdown_rect.centery - selected_txt.get_height() // 2))

        tri_center_x = dropdown_rect.right - 20
        tri_center_y = dropdown_rect.centery
        if dropdown_open:
            tri_points = [(tri_center_x - 6, tri_center_y + 3), (tri_center_x + 6, tri_center_y + 3), (tri_center_x, tri_center_y - 4)]
        else:
            tri_points = [(tri_center_x - 6, tri_center_y - 3), (tri_center_x + 6, tri_center_y - 3), (tri_center_x, tri_center_y + 4)]
        pygame.draw.polygon(screen, (230, 230, 230), tri_points)

        refresh_txt = dbgfont.render("Refresh", True, (255, 255, 255))
        screen.blit(
            refresh_txt,
            (refresh_btn.centerx - refresh_txt.get_width() // 2, refresh_btn.centery - refresh_txt.get_height() // 2),
        )

        folder_txt = dbgfont.render("Choose Folder", True, (255, 255, 255))
        force_txt = dbgfont.render(f"Force Native: {'ON' if force_native else 'OFF'}", True, (255, 255, 255))
        start_txt = dbgfont.render("Start", True, (255, 255, 255))

        screen.blit(folder_txt, (btn_folder.centerx - folder_txt.get_width() // 2, btn_folder.centery - folder_txt.get_height() // 2))
        screen.blit(force_txt, (force_native_btn.centerx - force_txt.get_width() // 2, force_native_btn.centery - force_txt.get_height() // 2))
        screen.blit(start_txt, (start_btn.centerx - start_txt.get_width() // 2, start_btn.centery - start_txt.get_height() // 2))

        if dropdown_open and eligible_dirs:
            list_x = dropdown_rect.x
            list_y = dropdown_rect.bottom + 6
            visible_count = min(dropdown_visible, len(eligible_dirs))
            list_h = visible_count * dropdown_item_h + 8

            # Opaque backdrop so dropdown content stays readable over all other UI text.
            backdrop = pygame.Rect(40, dropdown_rect.bottom + 2, WIDTH - 80, HEIGHT - dropdown_rect.bottom - 20)
            pygame.draw.rect(screen, (16, 16, 16), backdrop, border_radius=10)
            pygame.draw.rect(screen, (58, 58, 58), backdrop, width=1, border_radius=10)

            panel = pygame.Rect(list_x, list_y, dropdown_rect.width, list_h)
            pygame.draw.rect(screen, (22, 22, 22), panel, border_radius=8)
            pygame.draw.rect(screen, (112, 112, 112), panel, width=1, border_radius=8)

            for row in range(visible_count):
                idx = dropdown_scroll + row
                if idx >= len(eligible_dirs):
                    break
                item_rect = pygame.Rect(list_x + 4, list_y + 4 + row * dropdown_item_h, dropdown_rect.width - 8, dropdown_item_h)
                if idx == dropdown_selected:
                    pygame.draw.rect(screen, (72, 102, 82), item_rect, border_radius=5)
                elif item_rect.collidepoint(pygame.mouse.get_pos()):
                    pygame.draw.rect(screen, (62, 62, 62), item_rect, border_radius=5)

                item_text = dbgfont.render(_shorten_path(eligible_dirs[idx]["label"], max_len=76), True, (240, 240, 240))
                screen.blit(item_text, (item_rect.x + 8, item_rect.centery - item_text.get_height() // 2))

            if len(eligible_dirs) > dropdown_visible:
                scroll_hint = dbgfont.render("Use mouse wheel to scroll", True, (165, 165, 165))
                screen.blit(scroll_hint, (list_x + 10, list_y + list_h + 6))

        if not dropdown_open:
            folder_label = dbgfont.render(f"Folder: {_shorten_path(folder_path)}", True, (210, 210, 210))
            resolved_label = dbgfont.render(f"Resolved: {_shorten_path(resolved_folder)}", True, (210, 210, 210))
            file1_label = dbgfont.render(f"Native: {_shorten_path(native_file)}", True, (190, 220, 190) if native_file else (210, 210, 210))
            file2_label = dbgfont.render(f"System/Translated: {_shorten_path(translated_file)}", True, (190, 220, 190) if translated_file else (210, 210, 210))
            scan_label = dbgfont.render(f"Found: {len(eligible_dirs)} eligible folders", True, (190, 190, 190))
            screen.blit(scan_label, (70, 330))
            screen.blit(folder_label, (70, 358))
            screen.blit(resolved_label, (70, 386))
            screen.blit(file1_label, (70, 414))
            screen.blit(file2_label, (70, 442))

            hint = dbgfont.render("Use dropdown or manual choose, then click Start", True, (165, 165, 165))
            if native_file and translated_file and not force_native:
                mode = dbgfont.render("Mode: Dual transcript (native + system language)", True, (170, 220, 170))
                screen.blit(mode, (WIDTH // 2 - mode.get_width() // 2, 470))
            elif force_native and native_file:
                mode = dbgfont.render("Mode: Native only (forced)", True, (220, 210, 160))
                screen.blit(mode, (WIDTH // 2 - mode.get_width() // 2, 470))
            elif native_file or translated_file:
                mode = dbgfont.render("Mode: Single transcript", True, (220, 210, 160))
                screen.blit(mode, (WIDTH // 2 - mode.get_width() // 2, 470))

            if folder_path and not (native_file or translated_file):
                warn = dbgfont.render("Could not find a transcript file in that folder tree.", True, (230, 120, 120))
                screen.blit(warn, (WIDTH // 2 - warn.get_width() // 2, 470))
            if force_native and not native_file and (translated_file is not None):
                warn3 = dbgfont.render("Force native is ON but no native transcript was found.", True, (230, 170, 120))
                screen.blit(warn3, (WIDTH // 2 - warn3.get_width() // 2, 470))
            if not eligible_dirs:
                warn2 = dbgfont.render("Auto-scan found none. Use Choose Folder for manual selection.", True, (230, 170, 120))
                screen.blit(warn2, (WIDTH // 2 - warn2.get_width() // 2, 470))
            screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, 568))

        present_frame()
        clock.tick(60)

    return native_file, translated_file, resolved_folder, force_native


def unpack_lyrics(parent):
    global segments
    with open(f"{parent}/vocals_whisper_segments.json", "r") as f:
        json_data = json.load(f)
    segments = _build_segment_list(json_data)

    mixer.init()
    mixer.Sound(f"{parent}/{parent.split('/')[-1]}.mp3").play()  # program names the same


def _tokenize_for_render(seg_words):
    """
    Turn seg_words into render tokens that preserve leading spaces.
    Each token is (leading_space_count:int, text:str).
    """
    tokens = []
    for w in seg_words:
        s = w["word"]
        # Count leading spaces so we can advance x correctly (pygame can be weird w/ leading spaces)
        lead = 0
        while lead < len(s) and s[lead] == " ":
            lead += 1
        tokens.append((lead, s[lead:]))
    return tokens


def _render_highlighted_tokens(tokens, highlight_idx, x, y, max_width, partial_highlight=None):
    """
    Render tokens left-to-right. Highlight the token at highlight_idx.
    """
    x0 = x
    line_h = font.get_height()
    if line_h <= 0:
        line_h = 48
    current_line_h = line_h

    for idx, (lead_spaces, text) in enumerate(tokens):
        # advance for leading spaces
        if lead_spaces:
            space_w = font.size(" " * lead_spaces)[0]
            x += space_w

        if not text:
            continue

        base_surf = font.render(text, True, (255, 255, 255))
        base_w = base_surf.get_width()
        current_line_h = max(current_line_h, base_surf.get_height())
        # wrap (simple): if next word would exceed width, go to next line
        if x + base_w > x0 + max_width:
            x = x0
            y += current_line_h + 6
            current_line_h = base_surf.get_height()

        if partial_highlight and idx == partial_highlight.get("index"):
            n_chars = len(text)
            if "char_start" in partial_highlight and "char_end" in partial_highlight:
                i0 = max(0, min(n_chars, int(partial_highlight.get("char_start", 0))))
                i1 = max(i0, min(n_chars, int(partial_highlight.get("char_end", n_chars))))
            else:
                i0 = max(0, min(n_chars, int(n_chars * partial_highlight.get("start_frac", 0.0))))
                i1 = max(i0, min(n_chars, int(n_chars * partial_highlight.get("end_frac", 1.0))))
            pre = text[:i0]
            mid = text[i0:i1]
            post = text[i1:]

            pre_s = font.render(pre, True, (255, 255, 255))
            mid_s = font.render(mid, True, (255, 220, 80))
            post_s = font.render(post, True, (255, 255, 255))

            screen.blit(pre_s, (x, y))
            screen.blit(mid_s, (x + pre_s.get_width(), y))
            screen.blit(post_s, (x + pre_s.get_width() + mid_s.get_width(), y))
            x += pre_s.get_width() + mid_s.get_width() + post_s.get_width()
        else:
            color = (255, 220, 80) if idx == highlight_idx else (255, 255, 255)
            surf = font.render(text, True, color)
            screen.blit(surf, (x, y))
            x += surf.get_width()

    return y + current_line_h


def update_segment_view(ms, seg_list, y, state_name="orig"):
    """
    Generic updater for either original or translated track.
    Keeps indices advancing (no full rescans) and only retokenizes on segment change.
    """
    global seg_i, word_i, seg_i1, word_i1
    global _cached_seg_id, _cached_tokens, _cached_seg_id_1, _cached_tokens_1

    elapsed = ms / 1000.0
    if elapsed <= 1 or not seg_list:
        return y

    if state_name == "orig":
        si, wi = seg_i, word_i
        cache_id, cache_tokens = _cached_seg_id, _cached_tokens
    else:
        si, wi = seg_i1, word_i1
        cache_id, cache_tokens = _cached_seg_id_1, _cached_tokens_1

    def _persist_state():
        nonlocal si, wi, cache_id, cache_tokens
        if state_name == "orig":
            seg_i, word_i = si, wi
            _cached_seg_id, _cached_tokens = cache_id, cache_tokens
        else:
            seg_i1, word_i1 = si, wi
            _cached_seg_id_1, _cached_tokens_1 = cache_id, cache_tokens

    # Advance only once the next segment's words have actually started.
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
        _persist_state()
        return y

    seg = seg_list[si]
    words = seg.get("words") or []
    if not words:
        _persist_state()
        return y

    first_start = float(words[0]["start"])
    last_end = float(words[-1]["end"])
    if elapsed < first_start or elapsed > last_end:
        _persist_state()
        return y

    # advance word index inside the segment
    while wi < len(words) and elapsed >= words[wi]["end"]:
        wi += 1

    # clamp: if we're before first word start, keep wi at 0
    if wi < len(words) and elapsed < words[wi]["start"]:
        pass

    # retokenize only if segment changed
    seg_id = id(seg)
    if cache_id != seg_id:
        cache_tokens = _tokenize_for_render(words)
        cache_id = seg_id

    # render: show segment text (built from word tokens) + highlight current word
    # If wi == len(words), highlight nothing (segment basically finished)
    highlight_idx = wi if wi < len(words) else -1
    partial_highlight = None
    if wi < len(words):
        phone_segments = words[wi].get("phone_segments", [])
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
                partial_highlight = {
                    "index": wi,
                    "char_start": int(cur.get("char_start", 0)),
                    "char_end": int(cur.get("char_end", 0)),
                }
        else:
            phones = words[wi].get("phones", [])
            if phones:
                phone_idx = None
                for pi, p in enumerate(phones):
                    if p["start"] <= elapsed < p["end"]:
                        phone_idx = pi
                        break
                if phone_idx is None and elapsed >= phones[-1]["end"]:
                    phone_idx = len(phones) - 1
                if phone_idx is not None and len(phones) > 0:
                    partial_highlight = {
                        "index": wi,
                        "start_frac": phone_idx / len(phones),
                        "end_frac": (phone_idx + 1) / len(phones),
                    }

    bottom_y = _render_highlighted_tokens(
        cache_tokens,
        highlight_idx=highlight_idx,
        x=50,
        y=y,
        max_width=700,
        partial_highlight=partial_highlight,
    )

    if state_name == "orig":
        seg_i, word_i = si, wi
        _cached_seg_id, _cached_tokens = cache_id, cache_tokens
    else:
        seg_i1, word_i1 = si, wi
        _cached_seg_id_1, _cached_tokens_1 = cache_id, cache_tokens

    return bottom_y


def dispnotes(ms, dt_ms, x=50, bars_y=418, loop_row_y=556):
    _draw_note_strength_bars(ms, dt_ms, x=x, y=bars_y, width=700, height=102)

    source = analysis_note_strengths if analysis_note_strengths is not None else analysis_hpcp
    frame_index = _analysis_index_from_ms(ms, len(source)) if source is not None else -1
    cycle = _update_loop_tracker(frame_index, analysis_note_runs) if frame_index >= 0 else None
    if cycle is None:
        return

    # Loop readout lives on its own row below the bars + BPM line so it never
    # overlaps the note-strength panel. The progress ring sits at the left edge
    # (ahead of the label) to keep clear of the bottom-right credits/time text.
    pattern_text = cycle["motif_text"]
    if pattern_text:
        _draw_pattern_loader(cycle["progress"], x + 13, loop_row_y + 8)
        pat_label = dbgfont.render(f"Repeat~ [{pattern_text}]", True, (190, 245, 190))
        screen.blit(pat_label, (x + 34, loop_row_y))


def dispbeats(ms, x=50, y=530):
    global last_beat_found_at

    beat_visible = False
    if analysis_beat_times_ms is not None and len(analysis_beat_times_ms) > 0:
        beat_idx = int(np.searchsorted(analysis_beat_times_ms, ms, side="right") - 1)
        if beat_idx >= 0:
            last_beat_found_at = int(analysis_beat_times_ms[beat_idx])
            beat_visible = (ms - last_beat_found_at) < BEAT_VISUAL_MS
    elif analysis_beats is not None:
        beat_index = _analysis_index_from_ms(ms, len(analysis_beats))
        if beat_index >= 0 and analysis_beats[beat_index]:
            last_beat_found_at = ms
        beat_visible = (ms - last_beat_found_at) < BEAT_VISUAL_MS

    beat_label = "[  BEAT  ]" if beat_visible else "          "
    bpm_text = f"{analysis_bpm:5.1f}" if analysis_bpm is not None else "  n/a"
    label = dbgfont.render(f"BPM: {bpm_text} | {beat_label}", True, (255, 220, 80) if beat_visible else (200, 200, 200))
    screen.blit(label, (x, y))


def generarfrase():
    raise NotImplementedError


def load_lyrics_session(selected_native_file, selected_translated_file, resolved_folder, force_native):
    """Load transcript segments, audio, and analysis. Returns header surface for the session."""
    global segments, segments1, seg_i, word_i, seg_i1, word_i1
    global _cached_seg_id, _cached_tokens, _cached_seg_id_1, _cached_tokens_1
    global start_ticks, last_beat_found_at

    audio_anchor_file = selected_native_file or selected_translated_file

    if force_native and selected_native_file:
        segments = _load_segments_from_file(selected_native_file)
        segments1 = []
        mode_name = "Native only"
    elif selected_native_file and selected_translated_file:
        translated_json = _load_json_file(selected_translated_file)
        segments1 = _build_translated_segment_list(translated_json)
        if segments1 and _has_embedded_source_fields(translated_json):
            segments = _build_source_segment_list(translated_json)
            mode_name = "Dual transcript (embedded source)"
        elif segments1:
            native_path = _prefer_native_transcript_path(resolved_folder, selected_native_file)
            segments = _load_segments_from_file(native_path)
            mode_name = "Dual transcript"
        else:
            native_path = _prefer_native_transcript_path(resolved_folder, selected_native_file)
            segments = _load_segments_from_file(native_path)
            segments1 = []
            mode_name = "Single native (translation empty)"
    elif selected_native_file:
        native_path = _prefer_native_transcript_path(resolved_folder, selected_native_file)
        segments = _load_segments_from_file(native_path)
        segments1 = []
        mode_name = "Single native"
    else:
        translated_json = _load_json_file(selected_translated_file)
        segments1 = _build_translated_segment_list(translated_json)
        if segments1 and _has_embedded_source_fields(translated_json):
            segments = _build_source_segment_list(translated_json)
            mode_name = "Dual transcript (embedded source)"
        elif segments1:
            phone_native = _prefer_native_transcript_path(resolved_folder)
            if phone_native:
                segments = _load_segments_from_file(phone_native)
                mode_name = "Dual transcript"
            else:
                segments = _build_segment_list(translated_json)
                segments1 = []
                mode_name = "Single translated"
        else:
            segments = _build_segment_list(translated_json)
            segments1 = []
            mode_name = "Single translated"

    buildfor = dbgfont.render(
        f"{audio_anchor_file.split('/')[-2]} | {mode_name} | taeson.co",
        True,
        (255, 255, 255),
    )

    _start_audio_from_transcript_file(audio_anchor_file)
    _load_analysis_data(resolved_folder)

    seg_i = 0
    word_i = 0
    seg_i1 = 0
    word_i1 = 0
    _cached_seg_id = None
    _cached_tokens = None
    _cached_seg_id_1 = None
    _cached_tokens_1 = None
    start_ticks = pygame.time.get_ticks()
    last_beat_found_at = -1000

    return buildfor


def render_lyrics_frame(elapsed_ms, dt_ms, buildfor):
    """Draw one lyrics playback frame onto the global screen surface."""
    screen.fill((30, 30, 30))
    time_text = dbgfont.render(f"Elapsed: {elapsed_ms} ms", True, (255, 255, 255))
    credits = dbgfont.render("With <3 from Berkeley, Calif.", True, (190, 190, 190))
    screen.blit(
        credits,
        (
            screen.get_width() - credits.get_width(),
            screen.get_height() - credits.get_height() - time_text.get_height() - 2,
        ),
    )

    orig_bottom_y = update_segment_view(elapsed_ms, segments, y=50, state_name="orig")

    if segments1:
        base_trans_y = 140
        min_vertical_gap = 18
        trans_y = max(base_trans_y, orig_bottom_y + min_vertical_gap)
        update_segment_view(elapsed_ms, segments1, y=trans_y, state_name="trans")

    dispnotes(elapsed_ms, dt_ms)
    dispbeats(elapsed_ms)

    screen.blit(time_text, (780 - time_text.get_width(), 600 - time_text.get_height()))
    screen.blit(buildfor, (10, 8))


def run_lyrics_loop(buildfor, *, stop_check=None, present_frame=None):
    global running
    if stop_check is None:
        stop_check = lambda: False
    if present_frame is None:
        present_frame = pygame.display.flip

    running = True
    while running and not stop_check():
        dt = clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if stop_check():
            break

        elapsed_ms = _get_elapsed_ms()
        render_lyrics_frame(elapsed_ms, dt, buildfor)
        present_frame()


class PygameWorker(QObject):
    send_image = pyqtSignal(QImage)

    def _should_stop(self):
        return QThread.currentThread().isInterruptionRequested()

    def _present_frame(self):
        image = surface_to_qimage(screen)
        if not image.isNull():
            self.send_image.emit(image)

    def post_mouse_event(self, x, y, button=1, down=True):
        etype = pygame.MOUSEBUTTONDOWN if down else pygame.MOUSEBUTTONUP
        pygame.event.post(
            pygame.event.Event(etype, {"pos": (int(x), int(y)), "button": int(button)})
        )

    def post_wheel_event(self, x, y, delta_y):
        pygame.event.post(
            pygame.event.Event(
                pygame.MOUSEWHEEL,
                {"y": int(delta_y), "x": 0, "pos": (int(x), int(y))},
            )
        )

    def run(self):
        global _renderer_ready
        try:
            init_renderer()
            selected_native_file, selected_translated_file, resolved_folder, force_native = choose_generated_folder(
                stop_check=self._should_stop,
                present_frame=self._present_frame,
            )
            if not (selected_native_file or selected_translated_file) or self._should_stop():
                return

            buildfor = load_lyrics_session(
                selected_native_file,
                selected_translated_file,
                resolved_folder,
                force_native,
            )
            if self._should_stop():
                return

            run_lyrics_loop(
                buildfor,
                stop_check=self._should_stop,
                present_frame=self._present_frame,
            )
        finally:
            try:
                if mixer.get_init():
                    mixer.music.stop()
            except Exception:
                pass
            pygame.quit()
            _renderer_ready = False


class PygameDisplayLabel(QLabel):
    """Forwards mouse input from the Qt widget to the pygame event queue."""

    def __init__(self, worker: PygameWorker, parent=None):
        super().__init__(parent)
        self._worker = worker
        self.setMouseTracking(True)

    def _surface_pos(self, event: QMouseEvent):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None

        lw, lh = self.width(), self.height()
        pw, ph = pixmap.width(), pixmap.height()
        ox = max(0, (lw - pw) // 2)
        oy = max(0, (lh - ph) // 2)
        x = event.position().x() - ox
        y = event.position().y() - oy
        if x < 0 or y < 0 or x >= pw or y >= ph:
            return None
        sx = int(x * WIDTH / pw)
        sy = int(y * HEIGHT / ph)
        return max(0, min(WIDTH - 1, sx)), max(0, min(HEIGHT - 1, sy))

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = self._surface_pos(event)
        if pos is not None:
            pygame.event.post(
                pygame.event.Event(
                    pygame.MOUSEMOTION,
                    {"pos": pos, "rel": (0, 0), "buttons": (0, 0, 0)},
                )
            )
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        pos = self._surface_pos(event)
        if pos is not None:
            self._worker.post_mouse_event(pos[0], pos[1], button=event.button(), down=True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        pos = self._surface_pos(event)
        if pos is not None:
            self._worker.post_mouse_event(pos[0], pos[1], button=event.button(), down=False)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        pos = self._surface_pos(event)
        if pos is not None:
            delta = event.angleDelta().y()
            if delta:
                steps = delta // 120 if delta % 120 == 0 else (1 if delta > 0 else -1)
                self._worker.post_wheel_event(pos[0], pos[1], steps)
        super().wheelEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SONEX Lyrics")

        self.thread = QThread()
        self.worker = PygameWorker()
        self.worker.moveToThread(self.thread)

        self.label = PygameDisplayLabel(self.worker)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(DISPLAY_WIDTH, DISPLAY_HEIGHT)

        layout = QVBoxLayout()
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker.send_image.connect(self.update_image)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def update_image(self, image: QImage):
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.thread.requestInterruption()
        self.thread.quit()
        self.thread.wait(5000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())