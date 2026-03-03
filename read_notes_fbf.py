import numpy as np
import pygame
import time
import os

# --- 1. Load the Analysis Result ---
AUDIO = "22"
data_path = f"{AUDIO}_analysis.npz" #readfrom
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Run your analysis script first!")
    exit()

data = np.load(data_path)
hpcp_matrix = data['hpcp']
beat_map = data['beats']
bpm = data['bpm'][0]

pitch_classes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
THRESHOLD = 0.3
BEAT_VISUAL_MS = 100  # How long to keep the "BEAT" text visible

# --- 2. Initialize Audio ---
file_path = f"{AUDIO}/{AUDIO}.mp3"
pygame.mixer.init()
pygame.mixer.music.load(file_path)

print(f"File Loaded. Detected BPM: {bpm:.1f}")
print("Starting playback in 1 second... (Press Ctrl+C to stop)")
time.sleep(1)

pygame.mixer.music.play()
last_beat_found_at = -1000  # Track when the last beat occurred

# --- 3. The Playback Loop ---
try:
    while pygame.mixer.music.get_busy():
        # Get current playback position in milliseconds
        t = pygame.mixer.music.get_pos()
        
        if 0 <= t < len(hpcp_matrix):
            # A. Check for Beat
            if beat_map[t]:
                last_beat_found_at = t
            
            # Keep "BEAT" visible for a short window
            is_beat_visible = (t - last_beat_found_at) < BEAT_VISUAL_MS
            beat_label = "[  BEAT  ]" if is_beat_visible else "          "
            
            # B. Get Active Notes
            frame = hpcp_matrix[t]
            active = [pitch_classes[i] for i, v in enumerate(frame) if v >= THRESHOLD]
            note_str = ", ".join(active)
            
            # C. Print to Terminal
            # \r keeps everything on one line
            output = f"\rTime: {t:06d}ms | BPM: {bpm:5.1f} | {beat_label} | Notes: [{note_str}]"
            print(output.ljust(80), end="")
            
        # Sleep slightly to save CPU, but fast enough for 1ms precision
        time.sleep(0.002)

except KeyboardInterrupt:
    pygame.mixer.music.stop()
    print("\nPlayback stopped.")

print("\nFinished.")