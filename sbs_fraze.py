import pygame
from pygame import mixer
import json
import sys
import os

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "SONEX MULTILyricViewer_DBG LATIN-SPANISH Iniciado Sesión Como: Levi Brown ORC:0009-0007-5278-6761"
)

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

unpack_lyrics("generated_tested_audio/presiento")
unpack_lyrics_1("generated_tested_audio/presiento")
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