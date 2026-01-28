import argparse
import json
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QUrl, QRectF
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer


#python test_audio/gui_player.py --audio path/to/audio.mp3 --align vocals_whisper_segments_aligned.json

# ----------------------------
# Data model (from your JSON)
# ----------------------------

@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: List[Word]


class Alignment:
    """
    Loads a JSON with:
      {
        "segments": [
          {
            "start": <float>, "end": <float>, "text": <str>,
            "words": [{"word": <str>, "start": <float>, "end": <float>, ...}, ...]
          }, ...
        ]
      }
    """
    def __init__(self, segments: List[Segment]):
        self.segments = segments
        self._starts = [s.start for s in segments]

    @staticmethod
    def from_json(path: Path) -> "Alignment":
        data = json.loads(path.read_text(encoding="utf-8"))
        segs: List[Segment] = []
        for s in data.get("segments", []):
            words = []
            for w in s.get("words", []):
                # JSON uses key "word" for text
                wt = str(w.get("word", "")).strip()
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", ws))
                if wt:
                    words.append(Word(text=wt, start=ws, end=we))
            segs.append(
                Segment(
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                    text=str(s.get("text", "")).strip(),
                    words=words,
                )
            )

        # Sort by start time (stable view even if JSON has repeats)
        segs.sort(key=lambda x: (x.start, x.end))
        return Alignment(segs)

    def find_segment_index(self, t: float) -> Optional[int]:
        """
        Return an index for the segment that contains time t (start<=t<=end).
        If none contains it, return the nearest *next* segment index (or last).
        """
        if not self.segments:
            return None

        # Candidate: rightmost segment with start <= t
        i = bisect_right(self._starts, t) - 1

        # Check candidate and a small neighborhood (handles overlaps / edge cases)
        for j in (i, i - 1, i + 1):
            if 0 <= j < len(self.segments):
                s = self.segments[j]
                if s.start <= t <= s.end:
                    return j

        # If not inside any segment, choose next upcoming segment if possible
        next_i = bisect_right(self._starts, t)
        if next_i < len(self.segments):
            return next_i
        return len(self.segments) - 1

    def find_word_index(self, seg_idx: int, t: float) -> Optional[int]:
        s = self.segments[seg_idx]
        for k, w in enumerate(s.words):
            if w.start <= t <= w.end:
                return k
        return None


# ----------------------------
# Karaoke display widget
# ----------------------------

class KaraokeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.segment: Optional[Segment] = None
        self.current_word_idx: Optional[int] = None

        self.setMinimumWidth(520)
        self._font = QFont()
        self._font.setPointSize(28)
        self._font.setWeight(QFont.Weight.Medium)

        self._small_font = QFont()
        self._small_font.setPointSize(11)

        self._bg = QColor(18, 18, 20)
        self._fg = QColor(235, 235, 240)
        self._muted = QColor(160, 160, 170)
        self._highlight_bg = QColor(255, 220, 120, 200)
        self._highlight_fg = QColor(20, 20, 22)

    def set_state(self, segment: Optional[Segment], current_word_idx: Optional[int]):
        changed = (segment != self.segment) or (current_word_idx != self.current_word_idx)
        self.segment = segment
        self.current_word_idx = current_word_idx
        if changed:
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), self._bg)

        if not self.segment:
            p.setPen(self._muted)
            p.setFont(self._font)
            p.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignmentFlag.AlignCenter,
                       "No segment yet…\n(Press Play)")
            return

        # Header (time range)
        p.setFont(self._small_font)
        p.setPen(self._muted)
        header = f"{self.segment.start:.2f}s → {self.segment.end:.2f}s"
        p.drawText(QRectF(24, 16, self.width() - 48, 20), header)

        # Draw words with wrapping
        p.setFont(self._font)
        fm = QFontMetrics(self._font)

        x = 24.0
        y = 64.0
        max_w = self.width() - 48.0
        line_h = fm.height() + 16

        words = self.segment.words
        if not words:
            # Fallback to raw segment text if no word timings
            p.setPen(self._fg)
            p.drawText(QRectF(24, 64, max_w, self.height() - 96),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                       self.segment.text)
            return

        space_w = fm.horizontalAdvance(" ")

        for i, w in enumerate(words):
            ww = fm.horizontalAdvance(w.text)
            # Wrap if needed
            if x + ww > 24.0 + max_w:
                x = 24.0
                y += line_h

            rect = QRectF(x - 6, y - fm.ascent() - 6, ww + 12, fm.height() + 12)

            is_current = (self.current_word_idx == i)
            if is_current:
                p.fillRect(rect, self._highlight_bg)
                p.setPen(self._highlight_fg)
            else:
                p.setPen(self._fg)

            p.drawText(QRectF(x, y - fm.ascent(), ww, fm.height()), w.text)

            x += ww + space_w

        # Bottom hint
        p.setFont(self._small_font)
        p.setPen(self._muted)
        p.drawText(QRectF(24, self.height() - 28, self.width() - 48, 16),
                   "Space: Play/Pause   ←/→: Seek 5s   -/=: Offset")


# ----------------------------
# Main window
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self, audio_path: Path, align_path: Path, offset_sec: float):
        super().__init__()
        self.setWindowTitle("Karaoke Word Highlighter (PyQt6)")

        if not align_path.exists():
            raise FileNotFoundError(f"Alignment JSON not found: {align_path}")

        self.alignment = Alignment.from_json(align_path)
        self.offset_sec = offset_sec

        # Audio
        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        if audio_path.exists():
            self.player.setSource(QUrl.fromLocalFile(str(audio_path.resolve())))
        else:
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_playback_state)

        # UI
        self.karaoke = KaraokeWidget()

        self.context = QPlainTextEdit()
        self.context.setReadOnly(True)
        self.context.setMinimumWidth(340)
        self.context.setPlaceholderText("Upcoming / surrounding segments…")

        splitter = QSplitter()
        splitter.addWidget(self.karaoke)
        splitter.addWidget(self.context)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)

        self.open_btn = QPushButton("Open audio…")
        self.open_btn.clicked.connect(self.open_audio)

        self.time_label = QLabel("0.00s")
        self.offset_label = QLabel(f"offset: {self.offset_sec:+.2f}s")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self._slider_dragging = False

        controls = QHBoxLayout()
        controls.addWidget(self.open_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.time_label)
        controls.addWidget(self.offset_label)
        controls.addWidget(self.slider)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(splitter)
        root_layout.addLayout(controls)
        self.setCentralWidget(root)

        self._last_seg_idx: Optional[int] = None
        self.resize(1080, 560)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_playback_state(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setText("Pause")
        else:
            self.play_btn.setText("Play")

    def _on_duration(self, dur_ms: int):
        self.slider.setRange(0, max(0, dur_ms))

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        self.player.setPosition(self.slider.value())

    def _on_slider_moved(self, value: int):
        # preview time label while dragging
        sec = value / 1000.0
        self.time_label.setText(f"{sec:.2f}s")

    def _on_position(self, pos_ms: int):
        if not self._slider_dragging:
            self.slider.setValue(pos_ms)

        audio_t = pos_ms / 1000.0
        t = audio_t + self.offset_sec

        self.time_label.setText(f"{audio_t:.2f}s")
        self.offset_label.setText(f"offset: {self.offset_sec:+.2f}s")

        seg_idx = self.alignment.find_segment_index(t)
        if seg_idx is None:
            self.karaoke.set_state(None, None)
            return

        word_idx = self.alignment.find_word_index(seg_idx, t)
        self.karaoke.set_state(self.alignment.segments[seg_idx], word_idx)

        if seg_idx != self._last_seg_idx:
            self._last_seg_idx = seg_idx
            self._update_context(seg_idx)

    def _update_context(self, seg_idx: int):
        start = max(0, seg_idx - 4)
        end = min(len(self.alignment.segments), seg_idx + 6)
        lines: List[str] = []
        for i in range(start, end):
            s = self.alignment.segments[i]
            prefix = "▶ " if i == seg_idx else "  "
            lines.append(f"{prefix}{s.start:7.2f}–{s.end:7.2f}  {s.text}")
        self.context.setPlainText("\n".join(lines))

    def open_audio(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio Files (*.mp3 *.wav *.ogg *.flac);;All Files (*)",
        )
        if not file:
            return
        p = Path(file)
        if not p.exists():
            QMessageBox.warning(self, "Not found", "That file does not exist.")
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(p.resolve())))
        self.player.play()

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key.Key_Space:
            self.toggle_play()
            return

        if k == Qt.Key.Key_Left:
            self.player.setPosition(max(0, self.player.position() - 5000))
            return

        if k == Qt.Key.Key_Right:
            self.player.setPosition(self.player.position() + 5000)
            return

        # Offset tweak: '-' and '=' (US keyboard). Also support plus/minus on numpad.
        if k in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self.offset_sec -= 0.05
            self.offset_label.setText(f"offset: {self.offset_sec:+.2f}s")
            return

        if k in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
            self.offset_sec += 0.05
            self.offset_label.setText(f"offset: {self.offset_sec:+.2f}s")
            return

        super().keyPressEvent(event)


def parse_args() -> argparse.Namespace:
    default_align = Path("vocals_whisper_segments_aligned.json")
    # If you're running in the same environment as this chat, this path exists:
    sandbox_default = Path("/mnt/data/vocals_whisper_segments_aligned.json")
    if sandbox_default.exists():
        default_align = sandbox_default

    ap = argparse.ArgumentParser(description="Karaoke GUI: highlight current word from WhisperX-aligned JSON.")
    ap.add_argument("--audio", required=True, type=Path, help="Path to audio file (mp3/wav/ogg/…).")
    ap.add_argument("--align", default=default_align, type=Path, help="Alignment JSON with segments/words.")
    ap.add_argument("--offset", default=0.0, type=float, help="Seconds to add to audio time (sync adjustment).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    try:
        w = MainWindow(audio_path=args.audio, align_path=args.align, offset_sec=args.offset)
    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))
        return 1
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
