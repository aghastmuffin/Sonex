

from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt, QElapsedTimer, QTimer, QUrl
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import frase_core as core

WINDOW_TITLE = "🗣️ SONEX Lyrics, a TAESON.CO project."
FONT_PATH = os.path.join(os.path.dirname(__file__), "assets", "Darker Grotesque.ttf")
BRAND_DIR = os.path.join(os.path.dirname(__file__), "assets")

DARK_STYLE = """
QWidget {
    background-color: #1e1e1e;
    color: #f5f5f5;
}
QDialog {
    background-color: #181818;
}
QPushButton {
    background-color: #3a3a3a;
    border: 1px solid #5a5a5a;
    border-radius: 8px;
    padding: 8px 14px;
}
QPushButton:hover {
    background-color: #4a4a4a;
}
QPushButton:disabled {
    background-color: #373737;
    color: #888888;
}
QPushButton#startBtn:enabled {
    background-color: #46825f;
    border-color: #5a9a75;
}
QPushButton#forceNativeBtn:checked {
    background-color: #5a7864;
    border-color: #6a9a7a;
}
QComboBox {
    background-color: #343434;
    border: 1px solid #5a5a5a;
    border-radius: 8px;
    padding: 6px 10px;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QProgressBar {
    background-color: #373737;
    border: none;
    border-radius: 4px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0,
        stop:0 #5a78be, stop:1 #9ec8ff);
    border-radius: 4px;
}
QLabel#lyricsLabel {
    font-size: 36px;
    line-height: 1.25;
}
QLabel#statusOk {
    color: #bedcbe;
}
QLabel#statusWarn {
    color: #e6aa78;
}
QLabel#statusErr {
    color: #e67878;
}
"""


def load_app_font(size: int, bold: bool = False) -> QFont:
    if os.path.exists(FONT_PATH):
        fid = QFontDatabase.addApplicationFont(FONT_PATH)
        if fid >= 0:
            families = QFontDatabase.applicationFontFamilies(fid)
            if families:
                font = QFont(families[0], size)
                font.setBold(bold)
                return font
    font = QFont()
    font.setPointSize(size)
    font.setBold(bold)
    return font


def load_window_icon() -> QIcon:
    candidates = [
        "resolution-logo.png",
        "sonex-high-resolution-logo.png",
        "sonex-high-resolution-logo-transparent.png",
        "sonex-high-resolution-logo-grayscale.png",
        "sonex-high-resolution-logo-grayscale-transparent.png",
    ]
    for name in candidates:
        path = os.path.join(BRAND_DIR, name)
        if os.path.exists(path):
            icon = QIcon(path)
            if not icon.isNull():
                return icon
    return QIcon()


class LoopProgressWidget(QWidget):
    """Minimal arc widget — no native circular progress in Qt widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress = 0.0
        self.setFixedSize(26, 26)

    def set_progress(self, progress: float | None):
        self._progress = 0.0 if progress is None else max(0.0, min(1.0, float(progress)))
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() / 2, self.height() / 2
        radius = 10
        thickness = 3

        base_pen = QPen(QColor(90, 90, 90), thickness)
        painter.setPen(base_pen)
        painter.drawEllipse(int(cx - radius), int(cy - radius), radius * 2, radius * 2)

        if self._progress <= 0:
            return

        span = int(360 * 16 * self._progress)
        arc_pen = QPen(QColor(120, 255, 170), thickness)
        painter.setPen(arc_pen)
        rect = int(cx - radius), int(cy - radius), radius * 2, radius * 2
        painter.drawArc(*rect, 90 * 16, -span)


class NoteStrengthPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._bars: list[QProgressBar] = []
        self._labels: list[QLabel] = []

        for i, note in enumerate(core.pitch_classes):
            bar = QProgressBar()
            bar.setOrientation(Qt.Orientation.Vertical)
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(90)
            label = QLabel(note)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #dcdcdc; font-size: 11px;")
            self._bars.append(bar)
            self._labels.append(label)
            layout.addWidget(bar, 0, i)
            layout.addWidget(label, 1, i)

        self._na_label = QLabel("Note strength: [n/a]")
        self._na_label.setStyleSheet("color: #bebebe;")
        layout.addWidget(self._na_label, 0, 0, 2, len(core.pitch_classes))
        self._na_label.hide()

    def set_levels(self, levels):
        if levels is None:
            for bar in self._bars:
                bar.hide()
            for label in self._labels:
                label.hide()
            self._na_label.show()
            return

        self._na_label.hide()
        for bar, label, level in zip(self._bars, self._labels, levels):
            bar.show()
            label.show()
            bar.setValue(int(max(0.0, min(1.0, float(level))) * 100))


class FolderSelectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select generated folder")
        self.setMinimumWidth(760)

        self._eligible_dirs: list[dict] = []
        self._folder_path: str | None = None
        self._native_file: str | None = None
        self._translated_file: str | None = None
        self._resolved_folder: str | None = None
        self._force_native = False

        title = QLabel("Select generated folder")
        title.setFont(load_app_font(28))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scan_label = QLabel("Auto-scan eligible folders:")
        self.combo = QComboBox()
        self.combo.setMinimumHeight(40)

        self.refresh_btn = QPushButton("Refresh")
        self.choose_btn = QPushButton("Choose Folder")
        self.force_native_btn = QPushButton("Force Native: OFF")
        self.force_native_btn.setObjectName("forceNativeBtn")
        self.force_native_btn.setCheckable(True)

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setEnabled(False)

        row1 = QHBoxLayout()
        row1.addWidget(self.combo, stretch=1)
        row1.addWidget(self.refresh_btn)

        row2 = QHBoxLayout()
        row2.addWidget(self.choose_btn)
        row2.addWidget(self.force_native_btn)
        row2.addStretch()

        self.found_label = QLabel()
        self.folder_label = QLabel()
        self.resolved_label = QLabel()
        self.native_label = QLabel()
        self.translated_label = QLabel()
        self.mode_label = QLabel()
        self.hint_label = QLabel("Use dropdown or manual choose, then click Start")
        self.hint_label.setStyleSheet("color: #a5a5a5;")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for lbl in (
            self.found_label,
            self.folder_label,
            self.resolved_label,
            self.native_label,
            self.translated_label,
            self.mode_label,
        ):
            lbl.setWordWrap(True)

        buttons = QDialogButtonBox()
        buttons.addButton(self.start_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        cancel = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(scan_label)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.found_label)
        layout.addWidget(self.folder_label)
        layout.addWidget(self.resolved_label)
        layout.addWidget(self.native_label)
        layout.addWidget(self.translated_label)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.hint_label)
        layout.addWidget(buttons)

        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        self.refresh_btn.clicked.connect(self._refresh_dirs)
        self.choose_btn.clicked.connect(self._choose_folder)
        self.force_native_btn.toggled.connect(self._toggle_force_native)
        self.start_btn.clicked.connect(self._try_accept)
        cancel.clicked.connect(self.reject)

        self._refresh_dirs()

    def _apply_selection(self, index: int):
        if not (0 <= index < len(self._eligible_dirs)):
            return
        selection = self._eligible_dirs[index]
        self._folder_path = selection["folder"]
        self._native_file = selection.get("native")
        self._translated_file = selection.get("translated")
        self._resolved_folder = selection["folder"]
        self._update_status()

    def _on_combo_changed(self, index: int):
        self._apply_selection(index)

    def _refresh_dirs(self):
        self._eligible_dirs = core.refresh_eligible_dirs()
        self.combo.blockSignals(True)
        self.combo.clear()
        for item in self._eligible_dirs:
            self.combo.addItem(core.shorten_path(item["label"], max_len=78), item)
        self.combo.blockSignals(False)
        if self._eligible_dirs:
            self.combo.setCurrentIndex(0)
            self._apply_selection(0)
        else:
            self._native_file = None
            self._translated_file = None
            self._resolved_folder = None
            self._update_status()

    def _choose_folder(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose generated folder (or parent folder)",
            core.OUTPUT_ROOT or os.path.expanduser("~"),
        )
        if not selected:
            return
        self._folder_path = selected
        self._native_file, self._translated_file, self._resolved_folder = core._find_transcript_files(
            selected
        )
        self._update_status()

    def _toggle_force_native(self, checked: bool):
        self._force_native = checked
        self.force_native_btn.setText(f"Force Native: {'ON' if checked else 'OFF'}")
        self._update_status()

    def _can_start(self) -> bool:
        if self._force_native:
            return self._native_file is not None
        return (self._native_file is not None) or (self._translated_file is not None)

    def _update_status(self):
        self.found_label.setText(f"Found: {len(self._eligible_dirs)} eligible folders")
        self.folder_label.setText(f"Folder: {core.shorten_path(self._folder_path)}")
        self.resolved_label.setText(f"Resolved: {core.shorten_path(self._resolved_folder)}")

        native_color = "#bedcbe" if self._native_file else "#d2d2d2"
        trans_color = "#bedcbe" if self._translated_file else "#d2d2d2"
        self.native_label.setText(f"Native: {core.shorten_path(self._native_file)}")
        self.translated_label.setText(
            f"System/Translated: {core.shorten_path(self._translated_file)}"
        )
        self.native_label.setStyleSheet(f"color: {native_color};")
        self.translated_label.setStyleSheet(f"color: {trans_color};")

        mode_text = ""
        mode_color = "#d2d2d2"
        if self._native_file and self._translated_file and not self._force_native:
            mode_text = "Mode: Dual transcript (native + system language)"
            mode_color = "#aadca8"
        elif self._force_native and self._native_file:
            mode_text = "Mode: Native only (forced)"
            mode_color = "#dcc9a0"
        elif self._native_file or self._translated_file:
            mode_text = "Mode: Single transcript"
            mode_color = "#dcc9a0"
        elif self._folder_path and not (self._native_file or self._translated_file):
            mode_text = "Could not find a transcript file in that folder tree."
            mode_color = "#e67878"
        elif self._force_native and not self._native_file and self._translated_file:
            mode_text = "Force native is ON but no native transcript was found."
            mode_color = "#e6aa78"
        elif not self._eligible_dirs:
            mode_text = "Auto-scan found none. Use Choose Folder for manual selection."
            mode_color = "#e6aa78"

        self.mode_label.setText(mode_text)
        self.mode_label.setStyleSheet(f"color: {mode_color};")
        self.start_btn.setEnabled(self._can_start())

    def _try_accept(self):
        if self._can_start():
            self.accept()

    def selection(self):
        return self._native_file, self._translated_file, self._resolved_folder, self._force_native


class LyricsWindow(QMainWindow):
    def __init__(self, session: core.LyricsSession):
        super().__init__()
        self.session = session
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowIcon(load_window_icon())
        self.resize(900, 700)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        self.header_label = QLabel(session.header_text)
        self.header_label.setStyleSheet("color: #ffffff; font-size: 13px;")

        self.orig_lyrics = QLabel()
        self.orig_lyrics.setObjectName("lyricsLabel")
        self.orig_lyrics.setTextFormat(Qt.TextFormat.RichText)
        self.orig_lyrics.setWordWrap(True)
        self.orig_lyrics.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.orig_lyrics.setFont(load_app_font(36))

        self.trans_lyrics = QLabel()
        self.trans_lyrics.setObjectName("lyricsLabel")
        self.trans_lyrics.setTextFormat(Qt.TextFormat.RichText)
        self.trans_lyrics.setWordWrap(True)
        self.trans_lyrics.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.trans_lyrics.setFont(load_app_font(36))
        self.trans_lyrics.hide()

        self.note_panel = NoteStrengthPanel()

        beat_row = QHBoxLayout()
        self.bpm_label = QLabel("BPM:   n/a |           ")
        self.bpm_label.setStyleSheet("color: #c8c8c8; font-size: 13px;")
        beat_row.addWidget(self.bpm_label)
        beat_row.addStretch()

        loop_row = QHBoxLayout()
        self.loop_ring = LoopProgressWidget()
        self.loop_label = QLabel("")
        self.loop_label.setStyleSheet("color: #bef5be; font-size: 13px;")
        loop_row.addWidget(self.loop_ring)
        loop_row.addWidget(self.loop_label)
        loop_row.addStretch()
        self._loop_row_widget = QWidget()
        self._loop_row_widget.setLayout(loop_row)
        self._loop_row_widget.hide()

        footer = QHBoxLayout()
        self.credits_label = QLabel("With <3 from Berkeley, Calif.")
        self.credits_label.setStyleSheet("color: #bebebe; font-size: 12px;")
        self.time_label = QLabel("Elapsed: 0 ms")
        self.time_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        footer.addWidget(self.credits_label)
        footer.addStretch()
        footer.addWidget(self.time_label)

        root.addWidget(self.header_label)
        root.addWidget(self.orig_lyrics, stretch=2)
        root.addWidget(self.trans_lyrics, stretch=2)
        root.addWidget(self.note_panel)
        root.addLayout(beat_row)
        root.addWidget(self._loop_row_widget)
        root.addLayout(footer)

        if session.segments1:
            self.trans_lyrics.show()

        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        if session.audio_path:
            self._player.setSource(QUrl.fromLocalFile(session.audio_path))
            self._player.play()

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._frame_timer = QElapsedTimer()
        self._frame_timer.start()
        self._timer.start()

    def _elapsed_ms(self) -> int:
        pos = self._player.position()
        if pos >= 0:
            return int(pos)
        return 0

    def _tick(self):
        elapsed_ms = self._elapsed_ms()
        dt_ms = max(1, self._frame_timer.restart())

        orig_html = self.session.segment_html_at(elapsed_ms, track="orig")
        self.orig_lyrics.setText(
            f'<span style="color:#ffffff">{orig_html}</span>' if orig_html else ""
        )

        if self.session.segments1:
            trans_html = self.session.segment_html_at(elapsed_ms, track="trans")
            self.trans_lyrics.setText(
                f'<span style="color:#ffffff">{trans_html}</span>' if trans_html else ""
            )

        levels = self.session.update_note_bars(elapsed_ms, dt_ms)
        self.note_panel.set_levels(levels)

        bpm_text, beat_label, beat_visible = self.session.beat_display_at(elapsed_ms)
        color = "#ffdc50" if beat_visible else "#c8c8c8"
        self.bpm_label.setText(f"BPM: {bpm_text} | {beat_label}")
        self.bpm_label.setStyleSheet(f"color: {color}; font-size: 13px;")

        cycle = self.session.loop_display_at(elapsed_ms)
        if cycle and cycle.get("motif_text"):
            self._loop_row_widget.show()
            self.loop_ring.set_progress(cycle.get("progress"))
            self.loop_label.setText(f"Repeat~ [{cycle['motif_text']}]")
        else:
            self._loop_row_widget.hide()
            self.loop_ring.set_progress(0.0)
            self.loop_label.setText("")

        self.time_label.setText(f"Elapsed: {elapsed_ms} ms")

    def closeEvent(self, event):
        self._timer.stop()
        self._player.stop()
        super().closeEvent(event)


def generarfrase():
    raise NotImplementedError


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)

    dialog = FolderSelectDialog()
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return 0

    native, translated, folder, force_native = dialog.selection()
    if not (native or translated):
        return 0

    session = core.LyricsSession.load(native, translated, folder, force_native)
    if not session.audio_path:
        QMessageBox.warning(
            None,
            "Audio not found",
            "Could not find a matching .mp3 next to the transcript folder.",
        )

    window = LyricsWindow(session)
    window.show()

    print("job checking for MFA/fasterwhisper alignment accuracy")
    print("cleared for levi brown")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
