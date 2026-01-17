# Generate & Save MEL Spectrograms
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from audio.audiocontext import AudioContext


import librosa.display

def save_mel_spectrogram(
    ctx: AudioContext,
    output_name=None,
    n_mels=128,
    n_fft=2048,
    hop_length=512,
):
    """
    Converts audio to a Mel Spectrogram and saves it as a compressed .npz file.
    """
    S_dB = ctx.mel_spectrogram(
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    if output_name is None:
        base = os.path.splitext(os.path.basename(ctx.path))[0]
        output_name = f"{base}_mel.npz"

    np.savez_compressed(
        output_name,
        mel=S_dB,
        sr=ctx.sr,
        hop_length=hop_length,
        n_fft=n_fft,
        n_mels=n_mels,
    )

    return output_name


def render_mel(mel_spectrogram_path, output_image_path=None):
    """
    Renders a saved Mel Spectrogram .npz file to an image.
    """
    import matplotlib.pyplot as plt
    import librosa.display
    
    # Load the Mel Spectrogram data
    data = np.load(mel_spectrogram_path)
    S_dB = data['mel']
    sr = int(data['sr']) if 'sr' in data else None
    hop_length = int(data['hop_length']) if 'hop_length' in data else None
    
    # Create the plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', cmap='viridis', sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    
    # Save or show the image
    if output_image_path is None:
        output_image_path = os.path.splitext(mel_spectrogram_path)[0] + "_mel.png"
        
    plt.savefig(output_image_path)
    plt.close()
    return output_image_path

def render_mel_frames(mel_spectrogram_path, output_dir=None, frame_stride=1, max_frames=None, cmap='viridis', figsize=(3,6), show_colorbar=False, mode='imshow'):
    """
    USE FOR SLOW SYSTEMS, must be called manually
    Render one image per Mel spectrogram frame (time column).

    Parameters
    - mel_spectrogram_path: path to .npz created by save_mel_spectrogram
    - output_dir: directory to save frame images (created if missing)
    - frame_stride: render every Nth frame to thin out output (default 1)
    - max_frames: optional cap on number of frames to render
    - cmap: matplotlib colormap to use

    Returns
    - The output directory containing the frame images.
    """
    # TODO: integrate as setting
    import matplotlib.pyplot as plt
    import librosa.display

    data = np.load(mel_spectrogram_path)
    S_dB = data['mel']
    sr = int(data['sr']) if 'sr' in data else 44100
    hop_length = int(data['hop_length']) if 'hop_length' in data else 512

    if output_dir is None:
        output_dir = os.path.splitext(mel_spectrogram_path)[0] + "_frames"
    os.makedirs(output_dir, exist_ok=True)

    n_mels, n_frames = S_dB.shape
    times = librosa.frames_to_time(np.arange(n_frames+1), sr=sr, hop_length=hop_length)
    # Use global min/max so colors are consistent across frames
    vmin = float(S_dB.min())
    vmax = float(S_dB.max())

    rendered = 0
    for i in range(0, n_frames, frame_stride):
        if max_frames is not None and rendered >= max_frames:
            break
        # Take a single time column
        frame = S_dB[:, i:i+1]
        plt.figure(figsize=figsize)
        if mode == 'specshow':
            # specshow can struggle with single-column x-axis; we still provide consistent colors
            librosa.display.specshow(frame, y_axis='mel', cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            # Render with explicit extent so width maps to the exact frame's time span
            start_t = times[i]
            end_t = times[i+1]
            extent = [start_t, end_t, 0, n_mels]
            plt.imshow(frame, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
            plt.ylabel('Mel bins')
            plt.xlabel('Time (s)')
        if show_colorbar:
            plt.colorbar(format='%+2.0f dB')
        plt.title(f'Frame {i} ({times[i]:.3f}-{times[i+1]:.3f}s)')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'frame_{i:06d}_{times[i]:.3f}s.png')
        plt.savefig(out_path)
        plt.close()
        rendered += 1

    return output_dir

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.chdir("output")

    audio_file = "ADV.mp3"

    ctx = AudioContext(audio_file, sr=None)

    mel_path = save_mel_spectrogram(ctx)
    render_mel(mel_path)

    if input("Render Frame by Frame Mel too? ").strip().lower() in ("y", "yes"):
        render_mel_frames(mel_path, frame_stride=50)
