import pygame
from pygame import mixer
import json
import sys
import os
import subprocess
import numpy as np

pygame.init()
# Multi-Language GUI.
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "🗣️ SONEX MULTILyricViewer_DBG LATIN-SPANISH Iniciado Sesión Como: Levi Brown ORC:0009-0007-5278-6761"
)
print("DEPRECATION WARNING: sbs_fraze was introduced in SONEX pre-0.2-Berkeley, and is staged to be phased out of new features in pre-0.3-Berkeley, it will still be a vestigial part of the GUI until 1-PROD-Berkeley and is set for removal in 2-PROD-Davis. If you see this warning without doing anything out of the ordinary please report it on github.")
brand_dir = os.path.join(os.path.dirname(__file__), "gui", "assets", "brand")
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
    os.path.dirname(__file__), "gui", "assets", "brand", "Darker Grotesque.ttf"
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
analysis_beat_times_ms = None
analysis_frame_ms = 1.0
last_beat_found_at = -1000

# --- NEW: store broad segments instead of only flat words ---
segments = []     # original segments: [{start,end,text,words:[{start,end,word}]}]
segments1 = []    # translated segments

seg_i = 0
word_i = 0
seg_i1 = 0
word_i1 = 0

# cache so we only recompute tokens when the segment changes
_cached_seg_id = None
_cached_tokens = None

_cached_seg_id_1 = None
_cached_tokens_1 = None


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


def _build_source_segment_list(json_data):
    """Build source-language segments from embedded Argos source fields when present."""
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
                    }
                    for w in words
                    if "start" in w and "end" in w
                ],
            }
        )
    return out


def _load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def _has_embedded_source_fields(json_data):
    for seg in json_data:
        if seg.get("source_text"):
            return True
        src_words = seg.get("source_words")
        if isinstance(src_words, list) and len(src_words) > 0:
            return True
    return False


def _load_segments_from_file(file_path):
    json_data = _load_json_file(file_path)
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


def _find_transcript_files(folder_path):
    orig_candidates = [
        "vocals_whisper_segments.json",
        "mfa_vocals_phone_segments.json",
        "vocals_whisper_segments_aligned.json",
        "mfa_vocals_whisper_segments.json",
    ]
    trans_candidates = [
        "argos_translated.json",
        "translated.json",
        "vocals_whisper_segments_translated.json",
    ]

    def _resolve_in_dir(path):
        orig_path = next(
            (os.path.join(path, name) for name in orig_candidates if os.path.exists(os.path.join(path, name))),
            None,
        )
        trans_path = next(
            (os.path.join(path, name) for name in trans_candidates if os.path.exists(os.path.join(path, name))),
            None,
        )
        if orig_path or trans_path:
            return orig_path, trans_path, path
        return None, None, None

    direct_orig, direct_trans, direct_folder = _resolve_in_dir(folder_path)
    if direct_orig and direct_trans:
        return direct_orig, direct_trans, direct_folder

    partial_match = (direct_orig, direct_trans, direct_folder) if (direct_orig or direct_trans) else (None, None, None)
    for root, _, _ in os.walk(folder_path):
        nested_orig, nested_trans, nested_folder = _resolve_in_dir(root)
        if nested_orig and nested_trans:
            return nested_orig, nested_trans, nested_folder
        if (nested_orig or nested_trans) and not partial_match[2]:
            partial_match = (nested_orig, nested_trans, nested_folder)

    return partial_match


def _shorten_path(path, max_len=64):
    if not path:
        return "(not selected)"
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


def _find_analysis_file(folder_path, output_root=None):
    if not folder_path:
        return None

    folder_path = folder_path.rstrip("/")
    base_name = os.path.basename(folder_path)
    output_root = output_root or os.path.dirname(folder_path)

    preferred = [
        f"{base_name}_novocs_analysis.npz",
        f"{base_name}_vocs_analysis.npz",
        f"{base_name}_analysis.npz",
    ]

    for root in (output_root, folder_path):
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
    global analysis_hpcp, analysis_beats, analysis_bpm, analysis_beat_times_ms, analysis_frame_ms

    analysis_hpcp = None
    analysis_beats = None
    analysis_bpm = None
    analysis_beat_times_ms = None
    analysis_frame_ms = 1.0

    analysis_path = _find_analysis_file(folder_path)
    if not analysis_path:
        return

    try:
        data = np.load(analysis_path)
        analysis_hpcp = data["hpcp"] if "hpcp" in data else None
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
    except Exception:
        analysis_hpcp = None
        analysis_beats = None
        analysis_bpm = None
        analysis_beat_times_ms = None
        analysis_frame_ms = 1.0


def choose_generated_folder():
    folder_path = None
    file_1 = None
    file_2 = None
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
                        file_1, file_2, resolved_folder = _find_transcript_files(folder_path)
                elif start_btn.collidepoint(event.pos) and (file_1 or file_2):
                    choosing = False

        screen.fill((24, 24, 24))
        title = font.render("Select generated folder", True, (245, 245, 245))
        screen.blit(title, ((WIDTH - title.get_width()) // 2, 70))

        pygame.draw.rect(screen, (58, 58, 58), btn_folder, border_radius=10)
        pygame.draw.rect(
            screen,
            (70, 130, 95) if (file_1 or file_2) else (55, 55, 55),
            start_btn,
            border_radius=10,
        )

        folder_txt = dbgfont.render("Choose Folder", True, (255, 255, 255))
        start_txt = dbgfont.render("Start", True, (255, 255, 255))

        screen.blit(folder_txt, (btn_folder.centerx - folder_txt.get_width() // 2, btn_folder.centery - folder_txt.get_height() // 2))
        screen.blit(start_txt, (start_btn.centerx - start_txt.get_width() // 2, start_btn.centery - start_txt.get_height() // 2))

        folder_label = dbgfont.render(f"Folder: {_shorten_path(folder_path)}", True, (210, 210, 210))
        resolved_label = dbgfont.render(f"Resolved: {_shorten_path(resolved_folder)}", True, (210, 210, 210))
        file1_label = dbgfont.render(f"Transcript: {_shorten_path(file_1)}", True, (190, 220, 190) if file_1 else (210, 210, 210))
        file2_label = dbgfont.render(f"Legacy translated: {_shorten_path(file_2)}", True, (190, 220, 190) if file_2 else (210, 210, 210))
        screen.blit(folder_label, (70, 285))
        screen.blit(resolved_label, (70, 318))
        screen.blit(file1_label, (70, 351))
        screen.blit(file2_label, (70, 384))

        hint = dbgfont.render("Pick a folder, auto-find transcript, then click Start", True, (165, 165, 165))
        if file_1 and file_2:
            mode_hint = dbgfont.render("Mode: Embedded or legacy dual transcript", True, (170, 220, 170))
            screen.blit(mode_hint, (WIDTH // 2 - mode_hint.get_width() // 2, 545))
        elif file_1 or file_2:
            mode_hint = dbgfont.render("Mode: Single transcript", True, (220, 210, 160))
            screen.blit(mode_hint, (WIDTH // 2 - mode_hint.get_width() // 2, 545))
        elif folder_path:
            warn = dbgfont.render("Could not find a transcript file in that folder tree.", True, (230, 120, 120))
            screen.blit(warn, (WIDTH // 2 - warn.get_width() // 2, 545))
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, 515))

        pygame.display.flip()
        clock.tick(60)

    return file_1, file_2, resolved_folder


def unpack_lyrics(parent):
    global segments
    with open(f"{parent}/vocals_whisper_segments.json", "r") as f:
        json_data = json.load(f)
    segments = _build_segment_list(json_data)

    mixer.init()
    mixer.Sound(f"{parent}/{parent.split('/')[-1]}.mp3").play()  # program names the same


def unpack_lyrics_1(parent):
    global segments1
    with open(f"{parent}/argos_translated.json", "r") as f:
        json_data = json.load(f)
    segments1 = _build_segment_list(json_data)


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


def update_segment_view(ms, seg_list, y, state_name):
    """
    Generic updater for either original or translated track.
    Keeps indices advancing (no full rescans) and only retokenizes on segment change.
    """
    global seg_i, word_i, seg_i1, word_i1
    global _cached_seg_id, _cached_tokens, _cached_seg_id_1, _cached_tokens_1

    elapsed = ms / 1000.0
    if elapsed <= 1 or not seg_list:
        return y

    # choose which state vars/caches to use
    if state_name == "orig":
        si, wi = seg_i, word_i
        cache_id, cache_tokens = _cached_seg_id, _cached_tokens
    else:
        si, wi = seg_i1, word_i1
        cache_id, cache_tokens = _cached_seg_id_1, _cached_tokens_1

    # advance segment index if we're past its end
    while si < len(seg_list) and elapsed >= seg_list[si]["end"]:
        si += 1
        wi = 0
        cache_id = None
        cache_tokens = None

    if si >= len(seg_list):
        # write back state
        if state_name == "orig":
            seg_i, word_i = si, wi
            _cached_seg_id, _cached_tokens = cache_id, cache_tokens
        else:
            seg_i1, word_i1 = si, wi
            _cached_seg_id_1, _cached_tokens_1 = cache_id, cache_tokens
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

    # write back state
    if state_name == "orig":
        seg_i, word_i = si, wi
        _cached_seg_id, _cached_tokens = cache_id, cache_tokens
    else:
        seg_i1, word_i1 = si, wi
        _cached_seg_id_1, _cached_tokens_1 = cache_id, cache_tokens

    return bottom_y


def dispnotes(ms, x=50, y=520):
    if analysis_hpcp is None:
        label = dbgfont.render("Notes: [n/a]", True, (190, 190, 190))
        screen.blit(label, (x, y))
        return

    frame_index = _analysis_index_from_ms(ms, len(analysis_hpcp))
    if frame_index < 0:
        label = dbgfont.render("Notes: [n/a]", True, (190, 190, 190))
        screen.blit(label, (x, y))
        return

    frame = analysis_hpcp[frame_index]
    active_notes = [pitch_classes[i] for i, value in enumerate(frame) if value >= NOTE_THRESHOLD]
    note_text = ", ".join(active_notes) if active_notes else "-"
    label = dbgfont.render(f"Notes: [{note_text}]", True, (220, 220, 255))
    screen.blit(label, (x, y))


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


buildfor = dbgfont.render("SPANISH(C1) TESTING-LEVITAISUNKIMBROWN", True, (255, 255, 255))

selected_file_1, selected_file_2, resolved_folder = choose_generated_folder()
primary_file = selected_file_1 or selected_file_2
if not primary_file:
    pygame.quit()
    sys.exit()

primary_json = _load_json_file(primary_file)
segments = _build_source_segment_list(primary_json)
if _has_embedded_source_fields(primary_json):
    # Unified transcript file: translated words in `words`, original words in `source_words`.
    segments1 = _build_segment_list(primary_json)
elif selected_file_1 and selected_file_2 and selected_file_2 != primary_file:
    # Legacy dual-file mode fallback.
    segments1 = _load_segments_from_file(selected_file_2)
else:
    segments1 = []
_start_audio_from_transcript_file(primary_file)
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

    # --- NEW: update per segment, highlight current word inside it ---
    orig_bottom_y = update_segment_view(elapsed_ms, segments, y=50, state_name="orig")

    if segments1:
        base_trans_y = 140
        min_vertical_gap = 18
        trans_y = max(base_trans_y, orig_bottom_y + min_vertical_gap)
        update_segment_view(elapsed_ms, segments1, y=trans_y, state_name="trans")
    dispnotes(elapsed_ms)
    dispbeats(elapsed_ms)

    screen.blit(time_text, ((780 - time_text.get_width()), (600 - time_text.get_height())))
    screen.blit(buildfor, ((0 + buildfor.get_width()), (0 + buildfor.get_height())))
    pygame.display.flip()

pygame.quit()
sys.exit()