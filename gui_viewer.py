import pygame
from pygame import mixer
import json
import sys

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SONEX LyricViewer_DBG")

clock = pygame.time.Clock()
dbgfont = pygame.font.SysFont(None, 18)
font = pygame.font.SysFont(None, 48)
start_ticks = pygame.time.get_ticks()

running = True
lyrics = []
i = 0
def unpack_lyrics(parent):
    with open(f"{parent}/vocals_whisper_segments.json", "r") as f:
        json_data = json.load(f)
        f.close()
    for broad_chunk in json_data: #TODO: Update to show total text, broad chunk and then highlight the current word.
        for word in broad_chunk["words"]:
            start = word["start"]
            end = word["end"]
            word = word["word"]
            lyrics.append((start, end, word))
    mixer.init()
    mixer.Sound(f"{parent}/{parent}.mp3").play() #program names the same


def update_lyric(ms):
    global i
    elapsed = ms / 1000
    try:
        start = lyrics[i][0]
        end = lyrics[i][1]
        word = lyrics[i][2]

        if elapsed > 1:
            if start <= elapsed < end:
                lyric_text = font.render(f"{word}", True, (255, 255, 255))
                screen.blit(lyric_text, (50, 50))

            if elapsed >= end:
                i += 1

    except IndexError:
        pass



#unpack_lyrics("BEDROOM_Augustine_BloodOranges")
unpack_lyrics("RREGATON_ExperimentoMykeTowers")
while running:
    dt = clock.tick(60)  # dt = milliseconds since last frame

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    current_ticks = pygame.time.get_ticks()
    elapsed_ms = current_ticks - start_ticks

    screen.fill((30, 30, 30))

    time_text = dbgfont.render(f"Elapsed: {elapsed_ms} ms", True, (255, 255, 255))
    update_lyric(elapsed_ms)

    screen.blit(time_text, ((780 - time_text.get_width()), (600 - time_text.get_height())))
    pygame.display.flip()

pygame.quit()
sys.exit()