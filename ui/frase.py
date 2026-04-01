import pygame
from pygame import mixer
import json
import sys
import os
import numpy as np
from difflib import SequenceMatcher

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "🗣️ SONEX SingleViewer 0.3-b"
)

brand_dir = os.path.join(os.path.dirname(__file__),"assets")
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
font_path = os.path.join(
    os.path.dirname(__file__), "assets", "Darker Grotesque.ttf"
)
try:
    dbgfont = pygame.font.Font(font_path, 18)
    font = pygame.font.Font(font_path, 48)
except OSError:
    dbgfont = pygame.font.SysFont(None, 18)
    font = pygame.font.SysFont(None, 48)
start_ticks = pygame.time.get_ticks()

running = True

pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_THRESHOLD = 0.3
BEAT_VISUAL_MS = 100
TRANSCRIPT_FILE_CANDIDATES = [
    "vocals_whisper_segments.json",
    "mfa_vocals_phone_segments.json",
    "vocals_whisper_segments_aligned.json",
    "mfa_vocals_whisper_segments.json",
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
last_pattern_text = ""
note_bar_levels = np.zeros(12, dtype=np.float32)

# --- NEW: store broad segments instead of only flat words ---
segments = []     # original segments: [{start,end,text,words:[{start,end,word}]}]

seg_i = 0
word_i = 0

# cache so we only recompute tokens when the segment changes
_cached_seg_id = None
_cached_tokens = None


def default_output_root():
    app_name = "Sonex"
    if sys.platform.startswith("darwin"):
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    elif sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, app_name, "outputs")


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
                    }
                    for w in words
                ],
            }
        )
    return out


def _load_segments_from_file(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    return _build_segment_list(json_data)


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


def _resolve_transcript_in_dir(path):
    for name in TRANSCRIPT_FILE_CANDIDATES:
        candidate = os.path.join(path, name)
        if os.path.exists(candidate):
            return candidate, path
    return None, None


def _find_transcript_file(folder_path):
    direct_orig, direct_folder = _resolve_transcript_in_dir(folder_path)
    if direct_orig:
        return direct_orig, direct_folder

    for root, _, _ in os.walk(folder_path):
        nested_orig, nested_folder = _resolve_transcript_in_dir(root)
        if nested_orig:
            return nested_orig, nested_folder

    return None, None


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
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [
            d
            for d in dirs
            if d not in DISCOVERY_SKIP_DIR_NAMES
            and not d.startswith(".")
            and not d.startswith("_")
        ]

        files_set = set(files)
        match_name = next((name for name in TRANSCRIPT_FILE_CANDIDATES if name in files_set), None)
        if match_name is None:
            continue

        transcript_path = os.path.join(root, match_name)
        try:
            rel = os.path.relpath(root, search_root)
        except ValueError:
            rel = root
        if rel == ".":
            rel = os.path.basename(root)

        found.append(
            {
                "label": rel,
                "folder": root,
                "transcript": transcript_path,
            }
        )
        if len(found) >= max_results:
            break

    found.sort(key=lambda item: item["label"].lower())
    return found


def _find_analysis_file(folder_path):
    if not folder_path:
        return None

    base_name = os.path.basename(folder_path.rstrip("/"))
    preferred = [
        f"{base_name}_novocs_analysis.npz",
        f"{base_name}_vocs_analysis.npz",
        f"{base_name}_analysis.npz",
    ]

    for name in preferred:
        candidate = os.path.join(folder_path, name)
        if os.path.exists(candidate):
            return candidate

    try:
        for name in os.listdir(folder_path):
            if name.endswith("_analysis.npz"):
                return os.path.join(folder_path, name)
    except OSError:
        return None

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

    analysis_path = _find_analysis_file(folder_path)
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

        sample_rate_arr = data["sample_rate"] if "sample_rate" in data else None
        hop_size_arr = data["hop_size"] if "hop_size" in data else None
        if sample_rate_arr is not None and hop_size_arr is not None and len(sample_rate_arr) > 0 and len(hop_size_arr) > 0:
            sample_rate = float(sample_rate_arr[0])
            hop_size = float(hop_size_arr[0])
            if sample_rate > 0:
                analysis_frame_ms = (hop_size * 1000.0) / sample_rate

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
    top_k=4,
    half_life_ms=500,
    frame_ms=1.0,
    bridge_ms=400,
    min_run_ms=300,
    min_relative_strength=0.35,
):
    """
    Build stable note runs from noisy HPCP frames.

    Key changes vs original:
    - decay is derived from half_life_ms so it works correctly
      regardless of the hop size used during analysis.
    - Weights are 1/(rank+1) so a note consistently in 2nd place
      can beat one that only sporadically takes 1st.
    - min_relative_strength gates out weak notes so noise/percussion
      does not pollute the vote.
    - bridge and min_run are in ms, converted to frames internally.
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


def _pattern_text_for_ms(ms, runs, window_ms=5000, max_groups=8):
    if not runs:
        return ""

    start_ms = max(0, ms - window_ms)
    groups = []
    for note_idx, s, e in runs:
        if e < start_ms or s > ms:
            continue
        ov_s = max(s, start_ms)
        ov_e = min(e, ms)
        dur = ov_e - ov_s + 1
        if dur <= 0:
            continue

        # Convert duration to repeated-note width for readable pattern strings.
        reps = max(1, min(8, int(round(dur / 220.0))))
        groups.append(pitch_classes[int(note_idx)] * reps)

    if not groups:
        return ""
    return " ".join(groups[-max_groups:])


def _find_repeating_cycle_for_ms(
    frame_idx,
    runs,
    min_cycle_notes=6,
    max_cycle_notes=32,
    lookback_runs=80,
    min_similarity=0.62,
):
    """
    Detect a repeating note-order cycle around the current frame.
    Returns cycle metadata with progress, or None.

    frame_idx is the integer frame index (equal to elapsed_ms when frame_ms==1).
    The run list stores frame indices in start/end fields.
    """
    if not runs:
        return None

    cur_idx = None
    for i, (_n, s, e) in enumerate(runs):
        if s <= frame_idx <= e:
            cur_idx = i
            break
    if cur_idx is None:
        return None

    start_idx = max(0, cur_idx - lookback_runs + 1)
    seq = [int(r[0]) for r in runs[start_idx: cur_idx + 1]]
    rel_cur = len(seq) - 1

    best = None
    for cycle_len in range(min_cycle_notes, max_cycle_notes + 1):
        if rel_cur + 1 < cycle_len * 2:
            continue

        a = seq[rel_cur - (2 * cycle_len) + 1: rel_cur - cycle_len + 1]
        b = seq[rel_cur - cycle_len + 1: rel_cur + 1]
        if len(set(b)) < 2:
            continue

        sim = SequenceMatcher(None, a, b).ratio()
        if sim < min_similarity:
            continue

        if best is None or sim > best["sim"]:
            best = {
                "sim": sim,
                "cycle_len": cycle_len,
                "anchor_start": start_idx + (rel_cur - cycle_len + 1),
            }

    if best is None:
        return None

    cycle_len = best["cycle_len"]
    anchor_start = best["anchor_start"]
    cycle_start_idx = anchor_start + ((cur_idx - anchor_start) // cycle_len) * cycle_len
    cycle_end_idx = min(cycle_start_idx + cycle_len - 1, len(runs) - 1)

    cycle_start_ms = int(runs[cycle_start_idx][1])
    cycle_end_ms = int(runs[cycle_end_idx][2])
    cycle_dur = max(1, cycle_end_ms - cycle_start_ms + 1)
    progress = max(0.0, min(1.0, (frame_idx - cycle_start_ms) / float(cycle_dur)))

    motif = [
        pitch_classes[int(runs[i][0])]
        for i in range(cycle_start_idx, cycle_end_idx + 1)
    ]

    return {
        "motif_text": " ".join(motif),
        "progress": progress,
        "similarity": best["sim"],
    }


def _draw_pattern_loader(progress, center_x, center_y):
    """
    Draw a hollow circular loader that fills clockwise from a 0..1 progress value.
    """
    if progress is None:
        return
    progress = max(0.0, min(1.0, float(progress)))
    radius = 15
    thickness = 5

    # Base ring
    pygame.draw.circle(screen, (90, 90, 90), (center_x, center_y), radius, thickness)

    # Progress arc from top, clockwise
    start_angle = -np.pi / 2
    end_angle = start_angle + (2 * np.pi * progress)
    rect = pygame.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
    pygame.draw.arc(screen, (120, 255, 170), rect, start_angle, end_angle, thickness)


def choose_generated_folder():
    folder_path = None
    transcript_file = None
    resolved_folder = None

    repo_root = os.path.dirname(os.path.dirname(__file__))
    scan_roots = [default_output_root(), repo_root]

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
    start_btn = pygame.Rect(300, 420, 200, 64)

    def _apply_dropdown_selection(index):
        nonlocal folder_path, transcript_file, resolved_folder, dropdown_selected
        if not (0 <= index < len(eligible_dirs)):
            return
        dropdown_selected = index
        selection = eligible_dirs[index]
        folder_path = selection["folder"]
        transcript_file = selection["transcript"]
        resolved_folder = selection["folder"]

    if eligible_dirs:
        _apply_dropdown_selection(0)

    choosing = True
    while choosing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None

            if event.type == pygame.MOUSEWHEEL and dropdown_open and eligible_dirs:
                max_scroll = max(0, len(eligible_dirs) - dropdown_visible)
                dropdown_scroll = max(0, min(max_scroll, dropdown_scroll - event.y))

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                list_x = dropdown_rect.x
                list_y = dropdown_rect.bottom + 6
                list_h = min(dropdown_visible, len(eligible_dirs)) * dropdown_item_h + 8
                list_rect = pygame.Rect(list_x, list_y, dropdown_rect.width, list_h)

                if dropdown_rect.collidepoint(event.pos):
                    dropdown_open = bool(eligible_dirs) and not dropdown_open
                elif dropdown_open and eligible_dirs and list_rect.collidepoint(event.pos):
                    rel_y = event.pos[1] - (list_y + 4)
                    clicked = dropdown_scroll + (rel_y // dropdown_item_h)
                    if 0 <= clicked < len(eligible_dirs):
                        _apply_dropdown_selection(clicked)
                    dropdown_open = False
                elif refresh_btn.collidepoint(event.pos):
                    eligible_dirs = _refresh_eligible_dirs()
                    dropdown_open = False
                    dropdown_scroll = 0
                    if eligible_dirs:
                        _apply_dropdown_selection(0)
                    else:
                        dropdown_selected = -1
                elif btn_folder.collidepoint(event.pos):
                    selected = _pick_folder("Choose generated folder (or parent folder)")
                    if selected:
                        folder_path = selected
                        transcript_file, resolved_folder = _find_transcript_file(folder_path)
                    dropdown_open = False
                elif start_btn.collidepoint(event.pos) and transcript_file:
                    choosing = False
                else:
                    dropdown_open = False

        screen.fill((24, 24, 24))
        title = font.render("Select generated folder", True, (245, 245, 245))
        screen.blit(title, ((WIDTH - title.get_width()) // 2, 70))

        pygame.draw.rect(screen, (52, 52, 52), dropdown_rect, border_radius=8)
        pygame.draw.rect(screen, (90, 90, 90), dropdown_rect, width=1, border_radius=8)
        pygame.draw.rect(screen, (58, 58, 58), refresh_btn, border_radius=8)
        pygame.draw.rect(screen, (58, 58, 58), btn_folder, border_radius=10)
        pygame.draw.rect(
            screen,
            (70, 130, 95) if transcript_file else (55, 55, 55),
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
        start_txt = dbgfont.render("Start", True, (255, 255, 255))

        screen.blit(folder_txt, (btn_folder.centerx - folder_txt.get_width() // 2, btn_folder.centery - folder_txt.get_height() // 2))
        screen.blit(start_txt, (start_btn.centerx - start_txt.get_width() // 2, start_btn.centery - start_txt.get_height() // 2))

        if dropdown_open and eligible_dirs:
            list_x = dropdown_rect.x
            list_y = dropdown_rect.bottom + 6
            visible_count = min(dropdown_visible, len(eligible_dirs))
            list_h = visible_count * dropdown_item_h + 8
            panel = pygame.Rect(list_x, list_y, dropdown_rect.width, list_h)
            pygame.draw.rect(screen, (44, 44, 44), panel, border_radius=8)
            pygame.draw.rect(screen, (88, 88, 88), panel, width=1, border_radius=8)

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

        folder_label = dbgfont.render(f"Folder: {_shorten_path(folder_path)}", True, (210, 210, 210))
        resolved_label = dbgfont.render(f"Resolved: {_shorten_path(resolved_folder)}", True, (210, 210, 210))
        file1_label = dbgfont.render(f"Transcript: {_shorten_path(transcript_file)}", True, (190, 220, 190) if transcript_file else (210, 210, 210))
        scan_label = dbgfont.render(f"Found: {len(eligible_dirs)} eligible folders", True, (190, 190, 190))
        screen.blit(scan_label, (70, 330))
        screen.blit(folder_label, (70, 360))
        screen.blit(resolved_label, (70, 390))
        screen.blit(file1_label, (70, 420))

        hint = dbgfont.render("Use dropdown or manual choose, then click Start", True, (165, 165, 165))
        if folder_path and not transcript_file:
            warn = dbgfont.render("Could not find a transcript file in that folder tree.", True, (230, 120, 120))
            screen.blit(warn, (WIDTH // 2 - warn.get_width() // 2, 545))
        if not eligible_dirs:
            warn2 = dbgfont.render("Auto-scan found none. Use Choose Folder for manual selection.", True, (230, 170, 120))
            screen.blit(warn2, (WIDTH // 2 - warn2.get_width() // 2, 520))
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, 515))

        pygame.display.flip()
        clock.tick(60)

    return transcript_file, resolved_folder


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


def update_segment_view(ms, seg_list, y):
    """
    Generic updater for either original or translated track.
    Keeps indices advancing (no full rescans) and only retokenizes on segment change.
    """
    global seg_i, word_i
    global _cached_seg_id, _cached_tokens

    elapsed = ms / 1000.0
    if elapsed <= 1 or not seg_list:
        return y

    si, wi = seg_i, word_i
    cache_id, cache_tokens = _cached_seg_id, _cached_tokens

    # advance segment index if we're past its end
    while si < len(seg_list) and elapsed >= seg_list[si]["end"]:
        si += 1
        wi = 0
        cache_id = None
        cache_tokens = None

    if si >= len(seg_list):
        seg_i, word_i = si, wi
        _cached_seg_id, _cached_tokens = cache_id, cache_tokens
        return y

    seg = seg_list[si]
    words = seg["words"]

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

    seg_i, word_i = si, wi
    _cached_seg_id, _cached_tokens = cache_id, cache_tokens

    return bottom_y


def dispnotes(ms, dt_ms, x=50, y=520):
    global last_pattern_text

    _draw_note_strength_bars(ms, dt_ms, x=x, y=y - 102, width=700, height=102)

    source = analysis_note_strengths if analysis_note_strengths is not None else analysis_hpcp
    frame_index = _analysis_index_from_ms(ms, len(source)) if source is not None else -1
    cycle = _find_repeating_cycle_for_ms(frame_index, analysis_note_runs) if frame_index >= 0 else None
    if cycle is None:
        return

    pattern_text = cycle["motif_text"]

    # Anti-jitter smoothing: only freeze near-identical text.
    if pattern_text and last_pattern_text:
        ratio = SequenceMatcher(None, last_pattern_text, pattern_text).ratio()
        if ratio >= 0.90:
            pattern_text = last_pattern_text

    if pattern_text:
        last_pattern_text = pattern_text
        pat_label = dbgfont.render(f"Repeat~ [{pattern_text}]", True, (190, 245, 190))
        screen.blit(pat_label, (x, y - 22)) #TODO: Fix rendering here, move down to below the RPM/Beat counter
        _draw_pattern_loader(cycle["progress"], x + 710, y - 8)


def dispbeats(ms, x=50, y=545):
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



selected_file, resolved_folder = choose_generated_folder()
if not selected_file:
    pygame.quit()
    sys.exit()

buildfor = dbgfont.render(f"{selected_file.split('/')[-2]}, brought to you by taeson.co", True, (255, 255, 255))

segments = _load_segments_from_file(selected_file)
_start_audio_from_transcript_file(selected_file)
_load_analysis_data(resolved_folder)

seg_i = 0
word_i = 0
_cached_seg_id = None
_cached_tokens = None
start_ticks = pygame.time.get_ticks()
last_beat_found_at = -1000

print("job checking for MFA/fasterwhisper alignment accuracy")
print("cleared for levi brown")

while running:
    dt = clock.tick(60)  # dt = milliseconds since last frame

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    elapsed_ms = _get_elapsed_ms()

    screen.fill((30, 30, 30))
    time_text = dbgfont.render(f"Elapsed: {elapsed_ms} ms", True, (255, 255, 255))
    credits = dbgfont.render("With <3 from Berkeley, Calif.", True, (190, 190, 190))
    screen.blit(credits, ((screen.get_width() - credits.get_width()), screen.get_height() - credits.get_height() - time_text.get_height() - 2))

    # --- NEW: update per segment, highlight current word inside it ---
    update_segment_view(elapsed_ms, segments, y=50)
    dispnotes(elapsed_ms, dt)
    dispbeats(elapsed_ms)

    screen.blit(time_text, ((780 - time_text.get_width()), (600 - time_text.get_height())))
    screen.blit(buildfor, ((0 + buildfor.get_width()), (0 + buildfor.get_height())))
    pygame.display.flip()

pygame.quit()
sys.exit()