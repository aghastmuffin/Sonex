import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
y, sr = librosa.load("ojitos_lindos/htdemucs/ojitos_lindos/drums.mp3")

# Calculate RMS energy
rms = librosa.feature.rms(y=y)

# Plotting the RMS energy
plt.figure(figsize=(10, 4))
plt.plot(rms[0], label='RMS Energy')
plt.title('Audio Energy Over Time (RMS)')
plt.ylabel('Energy')
plt.xlabel('Frame')
plt.legend()
plt.show()
