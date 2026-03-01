#!/usr/bin/env python3
"""
Terminal-based audio player with synchronized transcription display.
Plays audio file and displays matching segments from output.json in real-time.
"""

import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import librosa
import sounddevice as sd
import numpy as np


class AudioSyncPlayer:
    def __init__(self, audio_path: str, json_path: str = "output.json"):
        """
        Initialize the audio sync player.
        
        Args:
            audio_path: Path to the audio file to play
            json_path: Path to the JSON file with transcription segments
        """
        self.audio_path = Path(audio_path)
        self.json_path = Path(json_path)
        
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # Load audio file
        print(f"📁 Loading audio: {self.audio_path}")
        self.audio_data, self.sr = librosa.load(str(self.audio_path), sr=None)
        self.duration = librosa.get_duration(y=self.audio_data, sr=self.sr)
        print(f"✓ Audio loaded: {self.duration:.2f}s @ {self.sr}Hz")
        
        # Load JSON segments
        print(f"📄 Loading transcription: {self.json_path}")
        with open(self.json_path) as f:
            data = json.load(f)
        self.segments = data.get("segments", [])
        print(f"✓ Loaded {len(self.segments)} segments")
        
        self.is_playing = False
        self.current_time = 0.0
        self.stream = None
        
    def get_current_segment(self) -> Optional[dict]:
        """Get the segment that matches the current playback time."""
        for segment in self.segments:
            if segment["start"] <= self.current_time <= segment["end"]:
                return segment
        return None
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio stream playback."""
        if status:
            print(f"⚠️  Audio stream status: {status}", file=sys.stderr)
        
        start_idx = int(self.current_time * self.sr)
        end_idx = start_idx + frames
        
        if start_idx >= len(self.audio_data):
            outdata.fill(0)
            self.is_playing = False
        else:
            chunk = self.audio_data[start_idx:end_idx]
            if len(chunk) < frames:
                chunk = np.pad(chunk, (0, frames - len(chunk)))
            outdata[:] = chunk.reshape(-1, 1)
        
        self.current_time += frames / self.sr
    
    def display_update(self):
        """Display current segment in a separate thread."""
        last_segment = None
        last_display_time = 0
        
        while self.is_playing:
            segment = self.get_current_segment()
            
            # Only display when segment changes or every 0.5 seconds
            if segment != last_segment or (time.time() - last_display_time) > 0.5:
                # Clear and display current info
                print(f"\r{'':80}", end="", flush=True)  # Clear line
                
                progress = (self.current_time / self.duration) * 100
                bar_length = 40
                filled = int(bar_length * self.current_time / self.duration)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                time_str = f"[{self.current_time:6.2f}s / {self.duration:6.2f}s]"
                
                if segment:
                    print(
                        f"\r{time_str} {bar} {progress:5.1f}%  |  {segment['text'][:50]}",
                        end="",
                        flush=True
                    )
                    if segment != last_segment:
                        last_segment = segment
                        # Print full segment info on a new line when it changes
                        print()
                        print(f"  📍 [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                        print(f"  💬 {segment['text']}")
                else:
                    print(f"\r{time_str} {bar} {progress:5.1f}%", end="", flush=True)
                
                last_display_time = time.time()
            
            time.sleep(0.01)
    
    def play(self):
        """Play the audio file with synchronized transcription display."""
        print("\n▶️  Starting playback...\n")
        self.is_playing = True
        self.current_time = 0.0
        
        # Start display update thread
        display_thread = threading.Thread(target=self.display_update, daemon=True)
        display_thread.start()
        
        try:
            # Create audio stream with callback
            with sd.OutputStream(
                samplerate=self.sr,
                channels=1,
                callback=self.audio_callback,
                blocksize=2048
            ):
                while self.is_playing:
                    time.sleep(0.01)
            
            print("\n✓ Playback complete!")
        
        except KeyboardInterrupt:
            print("\n⏹️  Playback interrupted by user")
        finally:
            self.is_playing = False
    
    def play_from_time(self, start_time: float):
        """Play from a specific time in seconds."""
        self.current_time = max(0, min(start_time, self.duration))
        print(f"⏩ Starting from {self.current_time:.2f}s")
        self.play()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Play audio with synchronized transcription display"
    )
    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to audio file (optional, will search common locations)"
    )
    parser.add_argument(
        "--json",
        default="output.json",
        help="Path to JSON transcription file (default: output.json)"
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0,
        help="Start playback at time in seconds (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Find audio file if not provided
    audio_file = args.audio
    if not audio_file:
        # Search common locations
        search_paths = [
            "RREGATON_ExperimentoMykeTowers/vocals.mp3",
            "test_audio/RREGATON_ExperimentoMykeTowers.mp3",
            "test_audio/ADV.mp3",
            "test_audio/ent sandman.mp3",
        ]
        
        for path in search_paths:
            if Path(path).exists():
                audio_file = path
                break
        
        if not audio_file:
            print("❌ No audio file found!")
            print("Please specify an audio file:")
            print("  python audio_sync_player.py <audio_file>")
            print("\nAvailable files:")
            for path in search_paths:
                full_path = Path(path)
                if full_path.exists():
                    print(f"  - {path}")
            sys.exit(1)
    
    try:
        player = AudioSyncPlayer(audio_file, args.json)
        
        if args.start > 0:
            player.play_from_time(args.start)
        else:
            player.play()
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
