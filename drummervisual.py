import pygame
import librosa
import numpy as np

# Load audio
AUDIO_PATH = "ojitos_lindos/htdemucs/ojitos_lindos/drums.mp3"
y, sr = librosa.load(AUDIO_PATH)

# Compute RMS energy
rms = librosa.feature.rms(y=y)[0]

# Get timing info
hop_length = 512
frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

# Init pygame
pygame.init()
pygame.mixer.init()

WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Audio Energy Visualizer")

clock = pygame.time.Clock()

# Load audio into pygame
pygame.mixer.music.load(AUDIO_PATH)

# Normalize RMS for display
rms_norm = rms / np.max(rms)

running = True
playing = False

font = pygame.font.SysFont(None, 36)

while running:
    screen.fill((20, 20, 20))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Press SPACE to play/pause
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not playing:
                    pygame.mixer.music.play()
                    playing = True
                else:
                    pygame.mixer.music.pause()
                    playing = False

    # Get current playback time
    if playing:
        current_time = pygame.mixer.music.get_pos() / 1000.0  # ms → sec
    else:
        current_time = 0

    # Find closest RMS frame
    idx = np.searchsorted(frame_times, current_time)
    idx = min(idx, len(rms_norm) - 1)

    energy = rms_norm[idx]

    # Draw energy bar
    bar_height = int(energy * HEIGHT * 0.8)
    pygame.draw.rect(
        screen,
        (0, 200, 255),
        (WIDTH // 2 - 50, HEIGHT - bar_height, 100, bar_height)
    )

    # Beat indicator (simple threshold)
    if energy > 0.6:
        text = font.render("BEAT!", True, (255, 80, 80))
        screen.blit(text, (WIDTH // 2 - 40, 50))

    # Instructions
    text = font.render("Press SPACE to Play/Pause", True, (200, 200, 200))
    screen.blit(text, (20, 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()