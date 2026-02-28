import pygame
from pygame import mixer
import json
import sys
import os
import subprocess

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "🗣️ SONEX MULTILyricViewer_DBG LATIN-SPANISH Iniciado Sesión Como: Levi Brown ORC:0009-0007-5278-6761"
)

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


def _pick_json_file(dialog_title):
    script = (
        f'set chosenFile to choose file with prompt "{dialog_title}" '
        'of type {"public.json"}\n'
        'POSIX path of chosenFile'
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


def _shorten_path(path, max_len=64):
    if not path:
        return "(not selected)"
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


def choose_transcript_files():
    file_1 = None
    file_2 = None

    btn1 = pygame.Rect(90, 200, 260, 56)
    btn2 = pygame.Rect(450, 200, 260, 56)
    start_btn = pygame.Rect(300, 420, 200, 64)

    choosing = True
    while choosing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn1.collidepoint(event.pos):
                    selected = _pick_json_file("Choose transcript file 1 (original)")
                    if selected:
                        file_1 = selected
                elif btn2.collidepoint(event.pos):
                    selected = _pick_json_file("Choose transcript file 2 (translated)")
                    if selected:
                        file_2 = selected
                elif start_btn.collidepoint(event.pos) and file_1 and file_2:
                    choosing = False

        screen.fill((24, 24, 24))
        title = font.render("Select transcript files", True, (245, 245, 245))
        screen.blit(title, ((WIDTH - title.get_width()) // 2, 70))

        pygame.draw.rect(screen, (58, 58, 58), btn1, border_radius=10)
        pygame.draw.rect(screen, (58, 58, 58), btn2, border_radius=10)
        pygame.draw.rect(
            screen,
            (70, 130, 95) if file_1 and file_2 else (55, 55, 55),
            start_btn,
            border_radius=10,
        )

        btn1_txt = dbgfont.render("Choose File 1", True, (255, 255, 255))
        btn2_txt = dbgfont.render("Choose File 2", True, (255, 255, 255))
        start_txt = dbgfont.render("Start", True, (255, 255, 255))

        screen.blit(btn1_txt, (btn1.centerx - btn1_txt.get_width() // 2, btn1.centery - btn1_txt.get_height() // 2))
        screen.blit(btn2_txt, (btn2.centerx - btn2_txt.get_width() // 2, btn2.centery - btn2_txt.get_height() // 2))
        screen.blit(start_txt, (start_btn.centerx - start_txt.get_width() // 2, start_btn.centery - start_txt.get_height() // 2))

        file1_label = dbgfont.render(f"File 1: {_shorten_path(file_1)}", True, (210, 210, 210))
        file2_label = dbgfont.render(f"File 2: {_shorten_path(file_2)}", True, (210, 210, 210))
        screen.blit(file1_label, (90, 285))
        screen.blit(file2_label, (90, 320))

        hint = dbgfont.render("Select both files, then click Start", True, (165, 165, 165))
        screen.blit(hint, (WIDTH // 2 - hint.get_width() // 2, 515))

        pygame.display.flip()
        clock.tick(60)

    return file_1, file_2


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


def _render_highlighted_tokens(tokens, highlight_idx, x, y, max_width):
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

        color = (255, 220, 80) if idx == highlight_idx else (255, 255, 255)
        surf = font.render(text, True, color)
        current_line_h = max(current_line_h, surf.get_height())
        # wrap (simple): if next word would exceed width, go to next line
        if x + surf.get_width() > x0 + max_width:
            x = x0
            y += current_line_h + 6
            current_line_h = surf.get_height()

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
    bottom_y = _render_highlighted_tokens(
        cache_tokens,
        highlight_idx=highlight_idx,
        x=50,
        y=y,
        max_width=700,
    )

    # write back state
    if state_name == "orig":
        seg_i, word_i = si, wi
        _cached_seg_id, _cached_tokens = cache_id, cache_tokens
    else:
        seg_i1, word_i1 = si, wi
        _cached_seg_id_1, _cached_tokens_1 = cache_id, cache_tokens

    return bottom_y


def generarfrase():
    raise NotImplementedError


buildfor = dbgfont.render("SPANISH(C1) TESTING-LEVITAISUNKIMBROWN", True, (255, 255, 255))

selected_file_1, selected_file_2 = choose_transcript_files()
if not selected_file_1 or not selected_file_2:
    pygame.quit()
    sys.exit()

segments = _load_segments_from_file(selected_file_1)
segments1 = _load_segments_from_file(selected_file_2)
_start_audio_from_transcript_file(selected_file_1)

seg_i = 0
word_i = 0
seg_i1 = 0
word_i1 = 0
_cached_seg_id = None
_cached_tokens = None
_cached_seg_id_1 = None
_cached_tokens_1 = None
start_ticks = pygame.time.get_ticks()

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
    orig_bottom_y = update_segment_view(elapsed_ms, segments, y=50, state_name="orig")

    base_trans_y = 140
    min_vertical_gap = 18
    trans_y = max(base_trans_y, orig_bottom_y + min_vertical_gap)
    update_segment_view(elapsed_ms, segments1, y=trans_y, state_name="trans")

    screen.blit(time_text, ((780 - time_text.get_width()), (600 - time_text.get_height())))
    screen.blit(buildfor, ((0 + buildfor.get_width()), (0 + buildfor.get_height())))
    pygame.display.flip()

pygame.quit()
sys.exit()