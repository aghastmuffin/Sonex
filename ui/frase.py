import pygame
from pygame import mixer
import json
import sys
import os
import subprocess
import numpy as np
from difflib import SequenceMatcher

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "🗣️ SONEX SINGLELyricViewer_DBG LATIN-SPANISH Iniciado Sesión Como: Levi Brown ORC:0009-0007-5278-6761"
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

analysis_hpcp = None
analysis_beats = None
analysis_bpm = None
analysis_note_strengths = None
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
    parent = os.path.dirname(file_path)
    parent_name = os.path.basename(parent)
    audio_path = os.path.join(parent, f"{parent_name}.mp3")
    if not os.path.exists(audio_path):
        return

    try:
        if not mixer.get_init():
            mixer.init()
        mixer.Sound(audio_path).play()
    except pygame.error:
        pass


def _pick_folder(dialog_title):
    script = (
        f'set chosenFolder to choose folder with prompt "{dialog_title}"\n'
        'POSIX path of chosenFolder'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _find_transcript_file(folder_path):
    orig_candidates = [
        "mfa_vocals_phone_segments.json",
        "vocals_whisper_segments.json",
        "vocals_whisper_segments_aligned.json",
        "mfa_vocals_whisper_segments.json",
    ]

    def _resolve_in_dir(path):
        orig_path = next((
            os.path.join(path, name) for name in orig_candidates if os.path.exists(os.path.join(path, name))
        ), None)
        if orig_path:
            return orig_path, path
        return None, None

    direct_orig, direct_folder = _resolve_in_dir(folder_path)
    if direct_orig:
        return direct_orig, direct_folder

    for root, _, _ in os.walk(folder_path):
        nested_orig, nested_folder = _resolve_in_dir(root)
        if nested_orig:
            return nested_orig, nested_folder

    return None, None


def _shorten_path(path, max_len=64):
    if not path:
        return "(not selected)"
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


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
    global analysis_hpcp, analysis_beats, analysis_bpm, analysis_note_strengths, analysis_note_runs, note_bar_levels

    analysis_hpcp = None
    analysis_beats = None
    analysis_bpm = None
    analysis_note_strengths = None
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
        bpm_arr = data["bpm"] if "bpm" in data else None
        if bpm_arr is not None and len(bpm_arr) > 0:
            analysis_bpm = float(bpm_arr[0])
        if analysis_hpcp is None and analysis_note_strengths is not None:
            analysis_hpcp = analysis_note_strengths
        if analysis_hpcp is not None and len(analysis_hpcp) > 0:
            analysis_note_runs = _extract_fuzzy_note_runs(analysis_hpcp)
    except Exception:
        analysis_hpcp = None
        analysis_beats = None
        analysis_bpm = None
        analysis_note_strengths = None
        analysis_note_runs = []
        note_bar_levels = np.zeros(12, dtype=np.float32)


def _draw_note_strength_bars(ms, dt_ms, x=50, y=420, width=700, height=120):
    global note_bar_levels

    source = analysis_note_strengths if analysis_note_strengths is not None else analysis_hpcp
    if source is None or ms < 0 or ms >= len(source):
        label = dbgfont.render("Note strength: [n/a]", True, (190, 190, 190))
        screen.blit(label, (x, y + height - 20))
        return

    frame = np.asarray(source[ms], dtype=np.float32)
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


def _extract_fuzzy_note_runs(hpcp, top_k=3, decay=0.93, bridge_frames=90, min_run_frames=120):
    """
    Build stable note runs from noisy HPCP frames.
    Fuzzy behavior:
    - Uses top-k weighted votes (not only strongest note)
    - Uses exponentially decayed state (temporal smoothing)
    - Bridges short interruptions between same-note runs
    """
    n = len(hpcp)
    if n == 0:
        return []

    state = np.zeros(12, dtype=np.float64)
    dominant = np.zeros(n, dtype=np.int16)
    rank_w = np.array([1.0, 0.72, 0.48], dtype=np.float64)

    for i in range(n):
        frame = np.asarray(hpcp[i], dtype=np.float64)
        state *= decay

        order = np.argsort(frame)[::-1]
        k = min(top_k, len(order))
        for r in range(k):
            idx = int(order[r])
            state[idx] += rank_w[r] * float(frame[idx])

        # continuity bonus to avoid jitter when top ranks swap rapidly
        if i > 0:
            state[int(dominant[i - 1])] += 0.06

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

    # Bridge: A ... B(short) ... A -> merge into A
    bridged = []
    i = 0
    while i < len(runs):
        if i + 2 < len(runs):
            a_note, a_s, a_e = runs[i]
            b_note, b_s, b_e = runs[i + 1]
            c_note, c_s, c_e = runs[i + 2]
            b_len = b_e - b_s + 1
            if a_note == c_note and b_len <= bridge_frames:
                bridged.append([a_note, a_s, c_e])
                i += 3
                continue
        bridged.append(runs[i])
        i += 1

    # Remove tiny runs by absorbing into neighbors when possible.
    compact = []
    for run in bridged:
        note, s, e = run
        run_len = e - s + 1
        if run_len < min_run_frames and compact:
            compact[-1][2] = e
        else:
            compact.append([note, s, e])

    return compact


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
    ms,
    runs,
    min_cycle_notes=3,
    max_cycle_notes=8,
    lookback_runs=28,
    min_similarity=0.84,
):
    """
    Detect a repeating note-order cycle around current time.
    Returns cycle metadata with progress only when a repeat is confirmed.
    """
    if not runs:
        return None

    cur_idx = None
    for i, (_n, s, e) in enumerate(runs):
        if s <= ms <= e:
            cur_idx = i
            break
    if cur_idx is None:
        return None

    start_idx = max(0, cur_idx - lookback_runs + 1)
    seq = [int(r[0]) for r in runs[start_idx:cur_idx + 1]]
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
    progress = (ms - cycle_start_ms) / float(cycle_dur)
    progress = max(0.0, min(1.0, progress))

    motif = [pitch_classes[int(runs[i][0])] for i in range(cycle_start_idx, cycle_end_idx + 1)]
    motif_text = " ".join(motif)

    return {
        "motif_text": motif_text,
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

    btn_folder = pygame.Rect(270, 200, 260, 56)
    start_btn = pygame.Rect(300, 420, 200, 64)

    choosing = True
    while choosing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_folder.collidepoint(event.pos):
                    selected = _pick_folder("Choose generated folder (or parent folder)")
                    if selected:
                        folder_path = selected
                        transcript_file, resolved_folder = _find_transcript_file(folder_path)
                elif start_btn.collidepoint(event.pos) and transcript_file:
                    choosing = False

        screen.fill((24, 24, 24))
        title = font.render("Select generated folder", True, (245, 245, 245))
        screen.blit(title, ((WIDTH - title.get_width()) // 2, 70))

        pygame.draw.rect(screen, (58, 58, 58), btn_folder, border_radius=10)
        pygame.draw.rect(
            screen,
            (70, 130, 95) if transcript_file else (55, 55, 55),
            start_btn,
            border_radius=10,
        )

        folder_txt = dbgfont.render("Choose Folder", True, (255, 255, 255))
        start_txt = dbgfont.render("Start", True, (255, 255, 255))

        screen.blit(folder_txt, (btn_folder.centerx - folder_txt.get_width() // 2, btn_folder.centery - folder_txt.get_height() // 2))
        screen.blit(start_txt, (start_btn.centerx - start_txt.get_width() // 2, start_btn.centery - start_txt.get_height() // 2))

        folder_label = dbgfont.render(f"Folder: {_shorten_path(folder_path)}", True, (210, 210, 210))
        resolved_label = dbgfont.render(f"Resolved: {_shorten_path(resolved_folder)}", True, (210, 210, 210))
        file1_label = dbgfont.render(f"Transcript: {_shorten_path(transcript_file)}", True, (190, 220, 190) if transcript_file else (210, 210, 210))
        screen.blit(folder_label, (70, 285))
        screen.blit(resolved_label, (70, 318))
        screen.blit(file1_label, (70, 351))

        hint = dbgfont.render("Pick a folder, auto-find transcript file, then click Start", True, (165, 165, 165))
        if folder_path and not transcript_file:
            warn = dbgfont.render("Could not find a transcript file in that folder tree.", True, (230, 120, 120))
            screen.blit(warn, (WIDTH // 2 - warn.get_width() // 2, 545))
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

    if analysis_hpcp is None or ms < 0 or ms >= len(analysis_hpcp):
        label = dbgfont.render("Notes: [n/a]", True, (190, 190, 190))
        screen.blit(label, (x, y))
        return

    frame = analysis_hpcp[ms]
    active_notes = [pitch_classes[i] for i, value in enumerate(frame) if value >= NOTE_THRESHOLD]
    note_text = ", ".join(active_notes[:3]) if active_notes else "-"
    label = dbgfont.render(f"Notes: [{note_text}]", True, (220, 220, 255))
    screen.blit(label, (x, y))

    cycle = _find_repeating_cycle_for_ms(ms, analysis_note_runs)
    if cycle is None:
        return

    pattern_text = cycle["motif_text"]
    if pattern_text and last_pattern_text:
        ratio = SequenceMatcher(None, last_pattern_text, pattern_text).ratio()
        if ratio >= 0.78:
            pattern_text = last_pattern_text

    if pattern_text:
        last_pattern_text = pattern_text
        pat_label = dbgfont.render(f"Repeat~ [{pattern_text}]", True, (190, 245, 190))
        screen.blit(pat_label, (x, y - 22))
        _draw_pattern_loader(cycle["progress"], x + 710, y - 8)


def dispbeats(ms, x=50, y=545):
    global last_beat_found_at

    beat_visible = False
    if analysis_beats is not None and 0 <= ms < len(analysis_beats):
        if analysis_beats[ms]:
            last_beat_found_at = ms
        beat_visible = (ms - last_beat_found_at) < BEAT_VISUAL_MS

    beat_label = "[  BEAT  ]" if beat_visible else "          "
    bpm_text = f"{analysis_bpm:5.1f}" if analysis_bpm is not None else "  n/a"
    label = dbgfont.render(f"BPM: {bpm_text} | {beat_label}", True, (255, 220, 80) if beat_visible else (200, 200, 200))
    screen.blit(label, (x, y))


def generarfrase():
    raise NotImplementedError


buildfor = dbgfont.render("SPANISH(C1) TESTING-LEVITAISUNKIMBROWN", True, (255, 255, 255))

selected_file, resolved_folder = choose_generated_folder()
if not selected_file:
    pygame.quit()
    sys.exit()

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

    current_ticks = pygame.time.get_ticks()
    elapsed_ms = current_ticks - start_ticks

    screen.fill((30, 30, 30))
    time_text = dbgfont.render(f"Elapsed: {elapsed_ms} ms", True, (255, 255, 255))

    # --- NEW: update per segment, highlight current word inside it ---
    update_segment_view(elapsed_ms, segments, y=50)
    dispnotes(elapsed_ms, dt)
    dispbeats(elapsed_ms)

    screen.blit(time_text, ((780 - time_text.get_width()), (600 - time_text.get_height())))
    screen.blit(buildfor, ((0 + buildfor.get_width()), (0 + buildfor.get_height())))
    pygame.display.flip()

pygame.quit()
sys.exit()