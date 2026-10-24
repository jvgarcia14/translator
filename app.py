import json
import queue
import re
import sys
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundcard as sc
import torch
import whisper
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget
)

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "config.json"

LANGUAGES = {
    "Auto-detect": "auto",
    "Mandarin / Chinese (Mainland & Taiwan)": "zh",
    "Indonesian / Bahasa Indonesia": "id",
    "English": "en",
    "Filipino / Tagalog": "tl",
    "Japanese": "ja",
    "Korean": "ko",
    "Thai": "th",
    "Vietnamese": "vi",
    "Malay": "ms",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Dutch": "nl",
    "Polish": "pl",
    "Ukrainian": "uk"
}


DEFAULT_CONFIG = {
    "audio": {
        "device_name_contains": "",
        "sample_rate": 16000,
        "frame_ms": 200,
        "window_seconds": 5.0,
        "step_seconds": 2.0,
        "silence_threshold": 0.008,
        "input_gain": 1.5,
        "target_rms": 0.08,
        "noise_gate": 0.003,
        "pre_emphasis": 0.95,
        "auto_gain": True,
        "max_auto_gain": 4.0,
        "limiter": 0.95
    },
    "whisper": {"language": "auto", "model": "small", "fp16": True},
    "overlay": {
        "width": 850, "height": 180, "x": 150, "y": 100,
        "font_family": "Segoe UI", "font_size": 22,
        "max_lines": 4, "opacity": 220
    }
}


def merge_config(defaults, saved):
    """Fill missing settings without discarding valid saved settings."""
    result = {}
    for key, default in defaults.items():
        value = saved.get(key, default)
        if isinstance(default, dict):
            result[key] = merge_config(default, value if isinstance(value, dict) else {})
        elif isinstance(value, type(default)) or (
            isinstance(default, float) and isinstance(value, (float, int))
        ):
            result[key] = value
        else:
            result[key] = default
    return result


def load_config():
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError("Configuration must be a JSON object")
            cfg = merge_config(DEFAULT_CONFIG, raw)
            if cfg != raw:
                save_config(cfg)
            return cfg
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            from datetime import datetime
            import shutil
            backup = CONFIG_PATH.with_name(
                f"config.invalid-{datetime.now():%Y%m%d-%H%M%S}.bak"
            )
            shutil.copy2(CONFIG_PATH, backup)
            print(f"Invalid config ({exc}). Backup saved to: {backup}")
    cfg = merge_config(DEFAULT_CONFIG, {})
    save_config(cfg)
    print(f"Created default configuration: {CONFIG_PATH}")
    return cfg


def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def language_name(code):
    for name, value in LANGUAGES.items():
        if value == code:
            return name
    return "Auto-detect"


class Signals(QObject):
    subtitle = Signal(str)
    status = Signal(str)
    level = Signal(float)


class SubtitleOverlay(QWidget):
    def __init__(self, cfg, signals):
        super().__init__()
        self.cfg = cfg
        self.drag_pos = None
        self.locked = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.container = QWidget()
        self.container.setObjectName("container")

        self.title_label = QLabel(
            f"LIVE TRANSLATION  •  {language_name(cfg['whisper']['language']).upper()} → ENGLISH"
        )
        self.title_label.setObjectName("title")

        self.status_label = QLabel("● Starting")
        self.status_label.setObjectName("status")

        top = QHBoxLayout()
        top.addWidget(self.title_label)
        top.addStretch()
        top.addWidget(self.status_label)

        self.subtitle_label = QLabel("Waiting for speech…")
        self.subtitle_label.setObjectName("subtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

        inner = QVBoxLayout(self.container)
        inner.setContentsMargins(20, 14, 20, 18)
        inner.setSpacing(10)
        inner.addLayout(top)
        inner.addWidget(self.subtitle_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.container)

        self.apply_style()
        self.resize(cfg["overlay"]["width"], cfg["overlay"]["height"])
        self.move(cfg["overlay"]["x"], cfg["overlay"]["y"])

        signals.subtitle.connect(self.subtitle_label.setText)
        signals.status.connect(lambda text: self.status_label.setText("● " + text))

    def apply_style(self):
        o = self.cfg["overlay"]
        self.subtitle_label.setFont(QFont(o["font_family"], o["font_size"]))
        self.setStyleSheet(f"""
            QWidget#container {{
                background-color: rgba(15, 18, 24, {o["opacity"]});
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 16px;
            }}
            QLabel#title {{
                color: rgba(255,255,255,150);
                font-family: "Segoe UI";
                font-size: 11px;
                font-weight: 700;
            }}
            QLabel#status {{
                color: #56d364;
                font-family: "Segoe UI";
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#subtitle {{
                color: #ffffff;
                font-family: "{o["font_family"]}";
                font-size: {o["font_size"]}px;
            }}
        """)

    def mousePressEvent(self, event):
        if not self.locked and event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, event):
        if (
            not self.locked
            and self.drag_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self.drag_pos)

    def mouseReleaseEvent(self, event):
        self.drag_pos = None


class SettingsDialog(QDialog):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Live Translator Settings")
        self.resize(560, 470)

        self.device = QComboBox()
        for device in sc.all_microphones(include_loopback=True):
            self.device.addItem(device.name)
        idx = self.device.findText(
            cfg["audio"]["device_name_contains"], Qt.MatchFlag.MatchContains
        )
        if idx >= 0:
            self.device.setCurrentIndex(idx)

        self.language = QComboBox()
        for name, code in LANGUAGES.items():
            self.language.addItem(name, code)
        idx = self.language.findData(cfg["whisper"]["language"])
        if idx >= 0:
            self.language.setCurrentIndex(idx)

        self.model = QComboBox()
        self.model.addItems(["tiny", "base", "small", "medium"])
        self.model.setCurrentText(cfg["whisper"]["model"])

        self.input_gain = QDoubleSpinBox()
        self.input_gain.setRange(0.5, 8.0)
        self.input_gain.setSingleStep(0.25)
        self.input_gain.setValue(cfg["audio"]["input_gain"])
        self.input_gain.setSuffix("×")

        self.target_rms = QDoubleSpinBox()
        self.target_rms.setRange(0.02, 0.20)
        self.target_rms.setSingleStep(0.01)
        self.target_rms.setDecimals(2)
        self.target_rms.setValue(cfg["audio"]["target_rms"])

        self.noise_gate = QDoubleSpinBox()
        self.noise_gate.setRange(0.001, 0.02)
        self.noise_gate.setSingleStep(0.001)
        self.noise_gate.setDecimals(4)
        self.noise_gate.setValue(cfg["audio"]["noise_gate"])

        self.font_size = QSpinBox()
        self.font_size.setRange(12, 36)
        self.font_size.setValue(cfg["overlay"]["font_size"])

        self.lines = QSpinBox()
        self.lines.setRange(2, 12)
        self.lines.setValue(cfg["overlay"]["max_lines"])

        form = QFormLayout()
        form.addRow("Audio device", self.device)
        form.addRow("Spoken language", self.language)
        form.addRow("Whisper model", self.model)
        form.addRow("Listener volume boost", self.input_gain)
        form.addRow("Auto-gain target", self.target_rms)
        form.addRow("Noise gate", self.noise_gate)
        form.addRow("Font size", self.font_size)
        form.addRow("Subtitle lines", self.lines)

        note = QLabel(
            "The volume boost is internal only. It makes the captured voice louder "
            "for transcription without changing what you hear in your headphones."
        )
        note.setWordWrap(True)
        note.setObjectName("note")

        save_button = QPushButton("Save and Start")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addStretch()
        layout.addLayout(buttons)

        self.setStyleSheet("""
            QDialog { background: #11151c; color: white; }
            QLabel { color: #d8dee9; font-size: 13px; }
            QLabel#note { color: #8b949e; padding-top: 8px; }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #1b2230; color: white;
                border: 1px solid #30394a;
                border-radius: 7px; padding: 8px;
            }
            QPushButton {
                background: #2f81f7; color: white; border: none;
                border-radius: 7px; padding: 10px 18px; font-weight: 600;
            }
            QPushButton:hover { background: #388bfd; }
        """)

    def apply(self):
        self.cfg["audio"]["device_name_contains"] = self.device.currentText()
        self.cfg["whisper"]["language"] = self.language.currentData()
        self.cfg["whisper"]["model"] = self.model.currentText()
        self.cfg["audio"]["input_gain"] = self.input_gain.value()
        self.cfg["audio"]["target_rms"] = self.target_rms.value()
        self.cfg["audio"]["noise_gate"] = self.noise_gate.value()
        self.cfg["overlay"]["font_size"] = self.font_size.value()
        self.cfg["overlay"]["max_lines"] = self.lines.value()
        save_config(self.cfg)


def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def is_duplicate(text, memory):
    current = normalize_text(text).lower()
    if not current:
        return True
    for old in memory:
        previous = normalize_text(old).lower()
        if current == previous:
            return True
        if SequenceMatcher(None, current, previous).ratio() >= 0.86:
            return True
    return False


def pick_device(cfg):
    target = cfg["audio"]["device_name_contains"].lower()
    devices = list(sc.all_microphones(include_loopback=True))
    for device in devices:
        if target and target in device.name.lower():
            return device
    speaker = sc.default_speaker()
    return sc.get_microphone(id=str(speaker.id), include_loopback=True)


def enhance_voice(audio, cfg):
    """Increase voice clarity internally without changing headphone volume."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio

    # Remove DC offset.
    audio = audio - float(np.mean(audio))

    # Pre-emphasis boosts consonants and speech detail.
    pre = float(cfg["audio"]["pre_emphasis"])
    emphasized = np.empty_like(audio)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - pre * audio[:-1]
    audio = emphasized

    # Fixed digital gain.
    audio *= float(cfg["audio"]["input_gain"])

    # Noise gate: suppress very quiet background noise.
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    if rms < float(cfg["audio"]["noise_gate"]):
        return np.zeros_like(audio)

    # Automatic gain control.
    if cfg["audio"].get("auto_gain", True):
        target = float(cfg["audio"]["target_rms"])
        max_gain = float(cfg["audio"]["max_auto_gain"])
        gain = min(max_gain, target / max(rms, 1e-6))
        audio *= gain

    # Soft limiter to avoid clipping after gain.
    limiter = float(cfg["audio"]["limiter"])
    audio = np.tanh(audio / max(limiter, 1e-6)) * limiter

    return audio.astype(np.float32)


def audio_worker(cfg, audio_queue, signals, stop_event):
    sample_rate = cfg["audio"]["sample_rate"]
    frames = int(sample_rate * cfg["audio"]["frame_ms"] / 1000)
    device = pick_device(cfg)
    signals.status.emit("Listening")
    print("Audio device:", device.name)

    with device.recorder(samplerate=sample_rate, channels=1) as recorder:
        while not stop_event.is_set():
            raw = recorder.record(numframes=frames).reshape(-1).astype(np.float32)
            data = enhance_voice(raw, cfg)

            while audio_queue.qsize() > 12:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                audio_queue.put(data, timeout=0.05)
            except queue.Full:
                pass


def transcribe_worker(cfg, audio_queue, signals, stop_event):
    sample_rate = cfg["audio"]["sample_rate"]
    window_samples = int(sample_rate * cfg["audio"]["window_seconds"])
    step_samples = int(sample_rate * cfg["audio"]["step_seconds"])
    threshold = cfg["audio"]["silence_threshold"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg["whisper"]["model"]
    signals.status.emit(f"Loading {model_name}")
    model = whisper.load_model(
        model_name, device=device, download_root=str(BASE / "models")
    )
    signals.status.emit("Live")

    audio_buffer = np.zeros(0, dtype=np.float32)
    samples_since_run = 0
    history = deque(maxlen=cfg["overlay"]["max_lines"])
    memory = deque(maxlen=12)

    configured_language = cfg["whisper"]["language"]
    whisper_language = None if configured_language == "auto" else configured_language

    while not stop_event.is_set():
        try:
            frame = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        audio_buffer = np.concatenate([audio_buffer, frame])
        samples_since_run += len(frame)

        if len(audio_buffer) > window_samples * 2:
            audio_buffer = audio_buffer[-window_samples * 2:]

        if len(audio_buffer) < window_samples or samples_since_run < step_samples:
            continue

        samples_since_run = 0
        chunk = audio_buffer[-window_samples:].copy()

        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        if rms < threshold:
            continue

        result = model.transcribe(
            chunk,
            language=whisper_language,
            task="translate",
            fp16=bool(cfg["whisper"]["fp16"] and device == "cuda"),
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=None,
            no_speech_threshold=0.55,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.2,
        )

        text = normalize_text(result.get("text") or "")
        if is_duplicate(text, memory):
            continue

        memory.append(text)
        history.append(text)
        signals.subtitle.emit("\n\n".join(history))


def main():
    cfg = load_config()
    app = QApplication(sys.argv)

    settings = SettingsDialog(cfg)
    if settings.exec() != QDialog.DialogCode.Accepted:
        return

    settings.apply()
    cfg = load_config()

    signals = Signals()
    overlay = SubtitleOverlay(cfg, signals)
    overlay.show()

    QShortcut(
        QKeySequence("F2"),
        overlay,
        activated=lambda: setattr(overlay, "locked", not overlay.locked),
    )
    QShortcut(QKeySequence("F3"), overlay, activated=overlay.hide)
    QShortcut(QKeySequence("F4"), overlay, activated=overlay.show)

    audio_queue = queue.Queue(maxsize=25)
    stop_event = threading.Event()

    threading.Thread(
        target=audio_worker,
        args=(cfg, audio_queue, signals, stop_event),
        daemon=True,
    ).start()

    threading.Thread(
        target=transcribe_worker,
        args=(cfg, audio_queue, signals, stop_event),
        daemon=True,
    ).start()

    exit_code = app.exec()
    stop_event.set()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
