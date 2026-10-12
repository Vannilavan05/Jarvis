"""Main Holographic Dashboard Widget for Jarvis.
Displays assistant state, audio input device, telemetry chips, and active command outcomes.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

try:
    from voice.audio_manager import VoiceState
except (ImportError, ValueError):
    from jarvis.voice.audio_manager import VoiceState


class DashboardPanel(QFrame):
    """Central status display telemetry panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "holo-panel")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("SYSTEM TELEMETRY")
        title.setProperty("class", "title")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(8)

        # Status Chip
        grid.addWidget(QLabel("Assistant State:"), 0, 0)
        self.lbl_state = QLabel("INITIALIZING")
        self.lbl_state.setProperty("class", "chip")
        grid.addWidget(self.lbl_state, 0, 1)

        # Wake Word
        grid.addWidget(QLabel("Wake Word:"), 0, 2)
        self.lbl_wake = QLabel("JARVIS")
        self.lbl_wake.setProperty("class", "chip")
        grid.addWidget(self.lbl_wake, 0, 3)

        # Input Audio Device
        grid.addWidget(QLabel("Audio Device:"), 1, 0)
        self.lbl_device = QLabel("Default Microphone")
        grid.addWidget(self.lbl_device, 1, 1)

        # Security Rank
        grid.addWidget(QLabel("Security Level:"), 1, 2)
        self.lbl_security = QLabel("PRIVILEGED")
        self.lbl_security.setProperty("class", "chip-active")
        grid.addWidget(self.lbl_security, 1, 3)

        layout.addLayout(grid)

        # Last Command Outcome
        layout.addWidget(QLabel("LATEST DIRECTIVE:"))
        self.lbl_last_cmd = QLabel("Awaiting voice command...")
        self.lbl_last_cmd.setStyleSheet("color: #00f0ff; font-weight: bold; padding: 4px; background: rgba(0,240,255,0.06); border-radius: 4px;")
        self.lbl_last_cmd.setWordWrap(True)
        layout.addWidget(self.lbl_last_cmd)

    def update_state(self, state: VoiceState):
        text = state.value.replace("_", " ")
        self.lbl_state.setText(text)
        if state in {VoiceState.LISTENING_FOR_COMMAND, VoiceState.LISTENING_FOR_WAKE}:
            self.lbl_state.setProperty("class", "chip-active")
        elif state in {VoiceState.ERROR, VoiceState.MUTED}:
            self.lbl_state.setProperty("class", "chip-alert")
        else:
            self.lbl_state.setProperty("class", "chip")
        self.lbl_state.style().unpolish(self.lbl_state)
        self.lbl_state.style().polish(self.lbl_state)

    def update_last_command(self, cmd_text: str, result_text: str, success: bool):
        color = "#00f0ff" if success else "#ff6080"
        self.lbl_last_cmd.setText(f"User: \"{cmd_text}\"\nJarvis: {result_text}")
        self.lbl_last_cmd.setStyleSheet(
            f"color: {color}; font-weight: bold; padding: 4px; background: rgba(0,240,255,0.06); border-radius: 4px;"
        )
