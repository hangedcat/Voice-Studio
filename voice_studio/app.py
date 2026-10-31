"""Voice Studio desktop window: Cover and Separate tabs (queue + Preview Mix), Combine tab and Train tab."""
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from .pipeline import ROOT, CoverSettings, StepFailed, cancel_running, combine, cover, list_voice_models, separate_song

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".mp4"}

# Analog console palette: warm charcoal panels, amber VU accent, signal-green for "done".
C = {
    "bg": "#16130f", "panel": "#201c17", "raised": "#2a251e", "line": "#3a3329",
    "text": "#ece3d3", "dim": "#9c907d", "amber": "#f0a33a", "amber_dim": "#7a5520",
    "green": "#8fc46a", "red": "#e2604a",
}


def stylesheet(display: str, mono: str) -> str:
    return f"""
    QWidget {{ background: {C['bg']}; color: {C['text']}; font-family: "{display}"; font-size: 13px; }}
    QTabWidget::pane {{ border: 1px solid {C['line']}; border-radius: 6px; top: -1px; background: {C['panel']}; }}
    QTabBar::tab {{ background: transparent; color: {C['dim']}; padding: 10px 22px; font-size: 14px;
                   letter-spacing: 2px; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {C['amber']}; border-bottom: 2px solid {C['amber']}; }}
    QFrame#strip {{ background: {C['raised']}; border: 1px solid {C['line']}; border-radius: 6px; }}
    QLabel#section {{ color: {C['amber']}; font-size: 11px; letter-spacing: 3px; background: transparent; }}
    QLabel#readout {{ font-family: "{mono}"; font-size: 22px; color: {C['amber']}; background: transparent; }}
    QLabel#hint, QLabel, QCheckBox, QFrame#strip QWidget#previewMix {{ background: transparent; }}
    QLabel#hint {{ color: {C['dim']}; font-size: 12px; }}
    QListWidget, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {C['bg']}; border: 1px solid {C['line']}; border-radius: 4px; padding: 5px;
        selection-background-color: {C['amber_dim']}; }}
    QListWidget {{ font-family: "{mono}"; font-size: 12px; padding: 4px; }}
    QListWidget::item {{ padding: 8px 6px; border-bottom: 1px solid {C['panel']}; }}
    QListWidget::item:selected {{ background: {C['raised']}; color: {C['amber']}; }}
    QPlainTextEdit {{ font-family: "{mono}"; font-size: 11px; color: {C['dim']}; }}
    QPushButton {{ background: {C['raised']}; border: 1px solid {C['line']}; border-radius: 4px;
                  padding: 8px 16px; letter-spacing: 1px; }}
    QPushButton:hover {{ border-color: {C['amber_dim']}; }}
    QPushButton:disabled {{ color: {C['line']}; }}
    QPushButton#go {{ background: {C['amber']}; color: {C['bg']}; border: none; font-size: 14px;
                     letter-spacing: 3px; padding: 12px; }}
    QPushButton#go:disabled {{ background: {C['amber_dim']}; color: {C['raised']}; }}
    QProgressBar {{ background: {C['bg']}; border: 1px solid {C['line']}; border-radius: 3px; height: 10px;
                   text-align: center; font-family: "{mono}"; font-size: 10px; color: {C['dim']}; }}
    QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {C['green']},
                          stop:0.75 {C['amber']}, stop:1 {C['red']}); border-radius: 2px; }}
    QSlider::groove:vertical {{ background: {C['bg']}; width: 6px; border-radius: 3px; border: 1px solid {C['line']}; }}
    QSlider::sub-page:vertical {{ background: {C['bg']}; }}
    QSlider::add-page:vertical {{ background: {C['amber_dim']}; border-radius: 3px; }}
    QSlider::handle:vertical {{ background: {C['text']}; height: 14px; margin: 0 -9px; border-radius: 2px; }}
    QSlider::groove:horizontal {{ background: {C['bg']}; height: 4px; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {C['amber']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {C['text']}; width: 10px; margin: -6px 0; border-radius: 5px; }}
    QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {C['line']}; border-radius: 3px; background: {C['bg']}; }}
    QCheckBox::indicator:checked {{ background: {C['amber']}; border-color: {C['amber']}; }}
    """


def section(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("section")
    return label


def strip(*widgets_or_layouts) -> QFrame:
    frame = QFrame()
    frame.setObjectName("strip")
    box = QVBoxLayout(frame)
    box.setContentsMargins(14, 12, 14, 14)
    for w in widgets_or_layouts:
        box.addLayout(w) if isinstance(w, (QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout)) else box.addWidget(w)
    return frame


class Job(QThread):
    """Runs one GPU job off the UI thread; fn receives (on_stage, on_epoch) callbacks."""
    stage = Signal(str)
    epoch = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.finished_ok.emit(self.fn(self.stage.emit, self.epoch.emit))
        except StepFailed as e:
            self.failed.emit(str(e))
        except Exception as e:  # surface anything unexpected instead of killing the thread silently
            self.failed.emit(f"{type(e).__name__}: {e}")


class SongQueue(QListWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.ElideMiddle)

    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e):
        self.files_dropped.emit([Path(u.toLocalFile()) for u in e.mimeData().urls()])


class PreviewMix(QFrame):
    """Two stems played together in sync, each on its own fader (name, default volume)."""

    def __init__(self, channels=(("Instrumental", 100), ("AI Vocals", 70))):
        super().__init__()
        self.setObjectName("previewMix")
        self.players = []
        self.faders = []
        for _ in range(2):
            player, out = QMediaPlayer(), QAudioOutput()
            player.setAudioOutput(out)
            self.players.append((player, out))

        grid = QGridLayout()
        for col, (name, default) in enumerate(channels):
            fader = QSlider(Qt.Vertical)
            fader.setRange(0, 100)
            fader.setFixedHeight(130)
            readout = QLabel()
            readout.setObjectName("readout")
            readout.setAlignment(Qt.AlignCenter)
            out = self.players[col][1]
            fader.valueChanged.connect(lambda v, o=out, r=readout: (o.setVolume(v / 100), r.setText(f"{v:3d}")))
            fader.setValue(default)
            self.faders.append(fader)
            label = QLabel(name)
            label.setObjectName("hint")
            grid.addWidget(readout, 0, col, Qt.AlignHCenter)
            grid.addWidget(fader, 1, col, Qt.AlignHCenter)
            grid.addWidget(label, 2, col, Qt.AlignHCenter)

        self.play = QPushButton("PLAY")
        self.play.clicked.connect(self.toggle)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.sliderMoved.connect(self.seek_to)
        self.clock = QLabel("0:00 / 0:00")
        self.clock.setObjectName("hint")
        transport = QHBoxLayout()
        transport.addWidget(self.play)
        transport.addWidget(self.seek, 1)
        transport.addWidget(self.clock)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(grid)
        box.addLayout(transport)

        self.lead = self.players[0][0]
        self.lead.durationChanged.connect(lambda d: (self.seek.setRange(0, d), self.tick()))
        self.timer = QTimer(self, interval=250, timeout=self.tick)
        self.set_tracks(None, None)

    def set_tracks(self, first: Path | None, second: Path | None):
        self.stop()
        for (player, _), path in zip(self.players, (first, second)):
            player.setSource(QUrl.fromLocalFile(str(path)) if path else QUrl())
        self.setEnabled(bool(first and second))

    def toggle(self):
        if self.lead.playbackState() == QMediaPlayer.PlayingState:
            self.stop()
        else:
            pos = self.lead.position()
            for player, _ in self.players:
                player.setPosition(pos)
                player.play()
            self.play.setText("PAUSE")
            self.timer.start()

    def stop(self):
        for player, _ in self.players:
            player.pause()
        self.play.setText("PLAY")
        self.timer.stop()

    def seek_to(self, ms: int):
        for player, _ in self.players:
            player.setPosition(ms)

    def tick(self):
        lead, follow = self.players[0][0], self.players[1][0]
        if abs(lead.position() - follow.position()) > 40:  # keep the two tracks in sync
            follow.setPosition(lead.position())
        if not self.seek.isSliderDown():
            self.seek.setValue(lead.position())
        fmt = lambda ms: f"{ms // 60000}:{ms // 1000 % 60:02d}"
        self.clock.setText(f"{fmt(lead.position())} / {fmt(lead.duration())}")
        if self.timer.isActive() and lead.playbackState() != QMediaPlayer.PlayingState:
            self.stop()


class QueueTab(QWidget):
    """A batch queue of Songs run one Job at a time, with a Preview Mix of the selected result.

    Subclasses supply the settings form and how to run, describe and preview one Song.
    """
    busy_changed = Signal(bool)

    title = ""
    channels = ()

    def __init__(self):
        super().__init__()
        self.job = None
        form = self.build_form()
        self.queue = SongQueue()
        self.queue.files_dropped.connect(self.add_songs)
        self.queue.currentItemChanged.connect(self.show_preview)
        hint = QLabel("Drop songs here, or")
        hint.setObjectName("hint")
        add = QPushButton("Add songs…")
        add.clicked.connect(self.pick_songs)
        clear = QPushButton("Clear finished")
        clear.clicked.connect(self.clear_finished)
        queue_bar = QHBoxLayout()
        queue_bar.addWidget(hint)
        queue_bar.addWidget(add)
        queue_bar.addStretch()
        queue_bar.addWidget(clear)

        self.go = QPushButton("RUN QUEUE")
        self.go.setObjectName("go")
        self.go.clicked.connect(self.run_next)

        self.preview = PreviewMix(self.channels)
        open_folder = QPushButton("Open output folder")
        open_folder.clicked.connect(self.open_output)

        left = QVBoxLayout()
        left.addWidget(section("Queue"))
        left.addWidget(self.queue, 1)
        left.addLayout(queue_bar)
        right = QVBoxLayout()
        right.addWidget(strip(section(self.title), form))
        right.addWidget(self.go)
        right.addWidget(strip(section("Preview mix"), self.preview, open_folder))
        right.addStretch()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)

    # --- per-tab hooks ---
    def build_form(self) -> QFormLayout:
        raise NotImplementedError

    def ready(self) -> bool:
        return True

    def job_fn(self, song: Path, out: Path):
        """Return fn(on_stage) that processes one Song into out and returns its result."""
        raise NotImplementedError

    def describe(self, result) -> str:
        return ""

    def tracks(self, result) -> tuple[Path, Path]:
        raise NotImplementedError

    # --- queue ---
    def pick_songs(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add songs", "", "Audio (*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.mp4)")
        self.add_songs([Path(f) for f in files])

    def add_songs(self, paths):
        for p in paths:
            if p.suffix.lower() in AUDIO_EXTS:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, {"song": p, "out": ROOT / "output" / p.stem, "status": "queued", "result": None})
                self.queue.addItem(item)
                self.render(item)

    def render(self, item):
        d = item.data(Qt.UserRole)
        marks = {"queued": "·", "running": "▶", "done": "●", "failed": "✕"}
        detail = {"done": lambda: self.describe(d["result"]), "running": lambda: d.get("stage", ""),
                  "failed": lambda: "failed — hover for details"}.get(d["status"], lambda: "")()
        item.setToolTip(d.get("error", ""))
        item.setText(f"{marks[d['status']]}  {d['song'].name:<40}  {detail}")
        item.setForeground(QColor({"done": C["green"], "failed": C["red"], "running": C["amber"]}.get(d["status"], C["text"])))

    def clear_finished(self):
        for i in reversed(range(self.queue.count())):
            if self.queue.item(i).data(Qt.UserRole)["status"] in ("done", "failed"):
                self.queue.takeItem(i)

    def run_next(self):
        if not self.ready():
            return
        item = next((self.queue.item(i) for i in range(self.queue.count())
                     if self.queue.item(i).data(Qt.UserRole)["status"] == "queued"), None)
        if item is None:
            self.set_busy(False)
            return
        self.set_busy(True)
        d = item.data(Qt.UserRole)
        d["status"] = "running"
        item.setData(Qt.UserRole, d)
        self.render(item)
        fn = self.job_fn(d["song"], d["out"])
        self.job = Job(lambda on_stage, _: fn(on_stage))
        self.job.stage.connect(lambda s: self.update_item(item, stage=f"{s}…"))
        self.job.finished_ok.connect(lambda r: (self.update_item(item, status="done", result=r), self.run_next()))
        self.job.failed.connect(lambda msg: (self.update_item(item, status="failed", error=msg), self.run_next()))
        self.job.start()

    def update_item(self, item, **changes):
        d = item.data(Qt.UserRole)
        d.update(changes)
        item.setData(Qt.UserRole, d)
        self.render(item)
        if item is self.queue.currentItem():
            self.show_preview(item)

    def show_preview(self, item, _previous=None):
        result = item.data(Qt.UserRole)["result"] if item else None
        self.preview.set_tracks(*(self.tracks(result) if result else (None, None)))

    def open_output(self):
        item = self.queue.currentItem()
        folder = item.data(Qt.UserRole)["out"] if item and item.data(Qt.UserRole)["result"] else ROOT / "output"
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def set_busy(self, busy: bool):
        self.go.setEnabled(not busy)
        self.go.setText("RUNNING…" if busy else "RUN QUEUE")
        self.busy_changed.emit(busy)


class CoverTab(QueueTab):
    title = "Conversion"
    channels = (("Instrumental", 100), ("AI Vocals", 70))

    def __init__(self):
        super().__init__()
        self.refresh_voices()

    def build_form(self) -> QFormLayout:
        self.voice = QComboBox()
        self.auto_pitch = QCheckBox("Auto (whole octaves)")
        self.auto_pitch.setChecked(True)
        self.pitch = QSpinBox()
        self.pitch.setRange(-24, 24)
        self.pitch.setValue(-12)
        self.pitch.setSuffix(" st")
        self.pitch.setEnabled(False)
        self.auto_pitch.toggled.connect(lambda on: self.pitch.setEnabled(not on))
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(self.auto_pitch)
        pitch_row.addWidget(self.pitch)
        defaults = CoverSettings()
        self.cleanup = QCheckBox("Remove reverb before converting")
        self.cleanup.setChecked(defaults.cleanup)
        self.f0 = QComboBox()
        self.f0.addItems(["fcpe", "rmvpe", "crepe"])
        self.f0.setCurrentText(defaults.f0_method)
        self.protect = QDoubleSpinBox(decimals=2, singleStep=0.05, maximum=0.5, value=defaults.protect)
        self.index_rate = QDoubleSpinBox(decimals=2, singleStep=0.1, maximum=1.0, value=defaults.index_rate)
        self.breaths = QCheckBox("Restore original breaths")
        self.breaths.setChecked(defaults.breaths)
        self.breath_level = QDoubleSpinBox(decimals=2, singleStep=0.1, maximum=2.0, value=defaults.breath_level)
        self.breath_level.setPrefix("level ")
        self.breath_level.setEnabled(defaults.breaths)
        self.breaths.toggled.connect(self.breath_level.setEnabled)
        breath_row = QHBoxLayout()
        breath_row.addWidget(self.breaths)
        breath_row.addWidget(self.breath_level)
        form = QFormLayout()
        form.addRow("Voice model", self.voice)
        form.addRow("Pitch", pitch_row)
        form.addRow("Cleanup", self.cleanup)
        form.addRow("Pitch tracker", self.f0)
        form.addRow("Breath protect", self.protect)
        form.addRow("Index rate", self.index_rate)
        form.addRow("Breaths", breath_row)
        return form

    def refresh_voices(self):
        current = self.voice.currentText()
        newest_first = sorted(list_voice_models(), key=lambda v: v.pth.stat().st_mtime, reverse=True)
        self.voices = {v.name: v for v in newest_first}
        self.voice.clear()
        self.voice.addItems(list(self.voices))
        if current in self.voices:
            self.voice.setCurrentText(current)

    def settings(self) -> CoverSettings:
        return CoverSettings(pitch=None if self.auto_pitch.isChecked() else self.pitch.value(),
                             cleanup=self.cleanup.isChecked(), f0_method=self.f0.currentText(),
                             protect=self.protect.value(), index_rate=self.index_rate.value(),
                             breaths=self.breaths.isChecked(), breath_level=self.breath_level.value())

    def ready(self) -> bool:
        if self.voice.currentText() not in self.voices:
            QMessageBox.warning(self, "No voice model", "Train or add a Voice Model first.")
            return False
        return True

    def job_fn(self, song, out):
        voice, settings = self.voices[self.voice.currentText()], self.settings()
        return lambda on_stage: cover(song, voice, out, settings, on_stage)

    def describe(self, result) -> str:
        return f"pitch {result.pitch:+d}"

    def tracks(self, result):
        return result.instrumental, result.ai_vocals


class SeparateTab(QueueTab):
    title = "Separation"
    channels = (("Instrumental", 100), ("Vocals", 100))

    def build_form(self) -> QFormLayout:
        self.clean = QCheckBox("Also save a no-reverb copy of the vocals")
        form = QFormLayout()
        form.addRow("Cleanup", self.clean)
        note = QLabel("Writes vocals.wav and instrumental.wav to output/<song name>.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        form.addRow(note)
        return form

    def job_fn(self, song, out):
        clean = self.clean.isChecked()
        return lambda on_stage: separate_song(song, out, clean, on_stage)

    def describe(self, result) -> str:
        return "vocals + instrumental" + (" + no reverb" if result.clean_acapella else "")

    def tracks(self, result):
        return result.instrumental, result.clean_acapella or result.acapella


class DropLine(QLineEdit):
    """A path field that also accepts a dropped file."""

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else e.ignore()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e):
        self.setText(e.mimeData().urls()[0].toLocalFile())
        self.editingFinished.emit()


class CombineTab(QWidget):
    """Mix two audio files: set the levels by ear in the Preview Mix, then export at those levels."""

    def __init__(self):
        super().__init__()
        self.job = None
        self.paths = []
        form = QFormLayout()
        for label in ("Track A", "Track B"):
            path = DropLine()
            path.setPlaceholderText("drop or browse an audio file")
            path.editingFinished.connect(self.load_preview)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _=False, p=path: self.pick_file(p))
            row = QHBoxLayout()
            row.addWidget(path, 1)
            row.addWidget(browse)
            form.addRow(label, row)
            self.paths.append(path)
        from_folder = QPushButton("Load song folder…")
        from_folder.clicked.connect(self.pick_song_folder)
        note = QLabel("Load song folder fills in instrumental.wav and the vocals from an output/<song> folder.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        form.addRow(from_folder)
        form.addRow(note)

        self.preview = PreviewMix((("Track A", 100), ("Track B", 100)))
        hint = QLabel("The mix is exported at these fader levels.")
        hint.setObjectName("hint")
        self.go = QPushButton("EXPORT MIX")
        self.go.setObjectName("go")
        self.go.clicked.connect(self.export)
        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        layout.addWidget(strip(section("Tracks"), form))
        layout.addWidget(strip(section("Preview mix"), self.preview, hint))
        layout.addWidget(self.go)
        layout.addWidget(self.status)
        layout.addStretch()

    def tracks(self) -> list[Path] | None:
        paths = [Path(p.text().strip().strip('"')) for p in self.paths]
        return paths if all(p.is_file() for p in paths) else None

    def pick_file(self, line: QLineEdit):
        file, _ = QFileDialog.getOpenFileName(self, "Pick audio", str(ROOT / "output"),
                                              "Audio (*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.mp4)")
        if file:
            line.setText(file)
            self.load_preview()

    def pick_song_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Song output folder", str(ROOT / "output"))
        if not folder:
            return
        folder = Path(folder)
        vocals = next((folder / n for n in ("ai_vocals.wav", "vocals_no_reverb.wav", "vocals.wav")
                       if (folder / n).exists()), None)
        if not (folder / "instrumental.wav").exists() or vocals is None:
            QMessageBox.warning(self, "Song folder", "That folder has no instrumental.wav and vocals to combine.")
            return
        self.paths[0].setText(str(folder / "instrumental.wav"))
        self.paths[1].setText(str(vocals))
        self.load_preview()

    def load_preview(self):
        tracks = self.tracks()
        self.preview.set_tracks(*(tracks or (None, None)))

    def export(self):
        tracks = self.tracks()
        if not tracks:
            QMessageBox.warning(self, "Combine", "Pick two existing audio files first.")
            return
        first = tracks[0]
        suggested = (first.parent / "mix.wav" if first.parent.parent == ROOT / "output"
                     else ROOT / "output" / f"{first.stem}_mix.wav")
        out, _ = QFileDialog.getSaveFileName(self, "Export mix", str(suggested),
                                             "WAV (*.wav);;MP3 (*.mp3);;FLAC (*.flac)")
        if not out:
            return
        gains = tuple(f.value() / 100 for f in self.preview.faders)
        self.preview.stop()
        self.go.setEnabled(False)
        self.status.setText("Exporting…")
        self.job = Job(lambda *_: combine(tracks[0], tracks[1], Path(out), gains))
        self.job.finished_ok.connect(lambda p: (self.go.setEnabled(True), self.status.setText(f"Saved {p}")))
        self.job.failed.connect(lambda msg: (self.go.setEnabled(True), self.status.setText(msg)))
        self.job.start()


class TrainTab(QWidget):
    busy_changed = Signal(bool)
    models_created = Signal()

    def __init__(self):
        super().__init__()
        self.job = None
        self.dataset = QLineEdit(str(ROOT / "datasets"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.pick_dataset)
        ds_row = QHBoxLayout()
        ds_row.addWidget(self.dataset, 1)
        ds_row.addWidget(browse)
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. my-voice-v3")
        self.epochs = QSpinBox(minimum=1, maximum=2000, value=200)
        self.save_every = QSpinBox(minimum=1, maximum=100, value=20)
        form = QFormLayout()
        form.addRow("Voice dataset", ds_row)
        form.addRow("Model name", self.name)
        form.addRow("Epochs", self.epochs)
        form.addRow("Save a model every", self.save_every)
        note = QLabel("Folder of dry, noise-free recordings of one voice (speaking + singing). "
                      "Every saved epoch becomes its own Voice Model so you can pick the best by ear.")
        note.setObjectName("hint")
        note.setWordWrap(True)

        self.go = QPushButton("START TRAINING")
        self.go.setObjectName("go")
        self.go.clicked.connect(self.start)
        self.stage = QLabel("idle")
        self.stage.setObjectName("readout")
        self.eta = QLabel("")
        self.eta.setObjectName("hint")
        self.progress = QProgressBar()
        self.progress.setFormat("epoch %v / %m")
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setMaximumBlockCount(500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        layout.addWidget(strip(section("Training job"), form, note))
        layout.addWidget(self.go)
        layout.addWidget(strip(section("Progress"), self.stage, self.progress, self.eta, self.log))

    def pick_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Voice dataset folder", self.dataset.text())
        if folder:
            self.dataset.setText(folder)

    def start(self):
        from .training import train_voice

        dataset, name = Path(self.dataset.text()), self.name.text().strip()
        if not dataset.is_dir() or not any(p.suffix.lower() in AUDIO_EXTS for p in dataset.iterdir()):
            QMessageBox.warning(self, "Voice dataset", "Pick a folder that contains your recordings.")
            return
        if not name or any(c in name for c in r'\/:*?"<>| '):
            QMessageBox.warning(self, "Model name", "Use a name without spaces or special characters.")
            return
        epochs, save_every = self.epochs.value(), self.save_every.value()
        log = ROOT / "output" / f"train_{name}.log"
        log.parent.mkdir(exist_ok=True)
        self.progress.setRange(0, epochs)
        self.progress.setValue(0)
        self.started = None
        self.set_busy(True)
        self.job = Job(lambda on_stage, on_epoch: train_voice(name, dataset, log, epochs, save_every,
                                                              on_stage=on_stage, on_epoch=on_epoch))
        self.job.stage.connect(lambda s: (self.stage.setText(s), self.log.appendPlainText(f"-> {s}")))
        self.job.epoch.connect(self.on_epoch)
        self.job.finished_ok.connect(self.done)
        self.job.failed.connect(self.failed)
        self.job.start()

    def on_epoch(self, epoch: int, total: int):
        if self.progress.value() == epoch:
            return  # Applio prints several lines per epoch
        now = time.monotonic()
        if self.started is None:
            self.started = (now, epoch)
        self.progress.setValue(epoch)
        t0, e0 = self.started
        if epoch > e0:
            left = (now - t0) / (epoch - e0) * (total - epoch)
            self.eta.setText(f"about {int(left // 3600)} h {int(left % 3600 // 60):02d} min left")

    def done(self, folders):
        self.set_busy(False)
        self.stage.setText("done")
        self.eta.setText("")
        self.log.appendPlainText("Voice models created:\n  " + "\n  ".join(f.name for f in folders))
        self.models_created.emit()

    def failed(self, msg: str):
        self.set_busy(False)
        self.stage.setText("failed")
        self.log.appendPlainText(msg)

    def set_busy(self, busy: bool):
        self.go.setEnabled(not busy)
        self.busy_changed.emit(busy)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Studio")
        self.resize(1100, 720)
        tabs = QTabWidget()
        self.cover, self.separate, self.combine, self.train = CoverTab(), SeparateTab(), CombineTab(), TrainTab()
        tabs.addTab(self.cover, "COVER")
        tabs.addTab(self.separate, "SEPARATE")
        tabs.addTab(self.combine, "COMBINE")
        tabs.addTab(self.train, "TRAIN")
        self.tabs = (self.cover, self.separate, self.combine, self.train)
        # one GPU job at a time (Combine is CPU-only, so it stays usable during long training)
        gpu_tabs = (self.cover, self.separate, self.train)
        for tab in gpu_tabs:
            others = [t for t in gpu_tabs if t is not tab]
            tab.busy_changed.connect(lambda b, others=others: [t.go.setEnabled(not b) for t in others])
        self.train.models_created.connect(self.cover.refresh_voices)
        self.setCentralWidget(tabs)

    def closeEvent(self, e):
        running = [t.job for t in self.tabs if t.job and t.job.isRunning()]
        if not running:
            return
        if QMessageBox.question(self, "Job running", "A job is still running. Stop it and quit?") != QMessageBox.Yes:
            e.ignore()
            return
        cancel_running()
        for job in running:
            job.wait(10000)


def pick_font(*families: str) -> str:
    available = set(QFontDatabase.families())
    return next((f for f in families if f in available), "Segoe UI")


def main():
    app = QApplication(sys.argv)
    display = pick_font("Bahnschrift", "Segoe UI Variable Display")
    mono = pick_font("Cascadia Mono", "Consolas")
    app.setFont(QFont(display, 10))
    app.setStyleSheet(stylesheet(display, mono))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
