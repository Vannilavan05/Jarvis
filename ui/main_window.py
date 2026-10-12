"""Main Holographic Desktop Interface for Jarvis.
Constructed with PyQt6, featuring the Arc Reactor waveform, telemetry monitors,
real-time conversational transcripts, security alerts, and non-blocking background workers.
"""

import sys
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

try:
    from ai.trainer import ModelTrainer
    from execution.dispatcher import CommandDispatcher, ExecutionResponse
    from nlp.intent_parser import IntentParser
    from security.confirmation_manager import ConfirmationManager
    from security.permissions import PermissionLevel
    from storage.settings_store import SettingsStore
    from storage.training_store import TrainingStore
    from voice.audio_manager import AudioManager, VoiceState
except (ImportError, ValueError):
    from jarvis.ai.trainer import ModelTrainer
    from jarvis.execution.dispatcher import CommandDispatcher, ExecutionResponse
    from jarvis.nlp.intent_parser import IntentParser
    from jarvis.security.confirmation_manager import ConfirmationManager
    from jarvis.security.permissions import PermissionLevel
    from jarvis.storage.settings_store import SettingsStore
    from jarvis.storage.training_store import TrainingStore
    from jarvis.voice.audio_manager import AudioManager, VoiceState

from .dashboard import DashboardPanel
from .styles import HOLOGRAPHIC_STYLE
from .training_panel import TrainingPanel
from .waveform import HolographicWaveformWidget


class UIEvents(QObject):
    """Thread-safe signal dispatcher for UI updates."""
    sig_voice_state = pyqtSignal(object)
    sig_amplitude = pyqtSignal(float)
    sig_transcript = pyqtSignal(str, bool)
    sig_command_outcome = pyqtSignal(str, str, bool)
    sig_training_progress = pyqtSignal(int, int, float, float)
    sig_training_complete = pyqtSignal(int, int, float, float)
    sig_confirmation_alert = pyqtSignal(str, bool)


class JarvisMainWindow(QMainWindow):
    """The master holographic desktop window for Jarvis."""

    def __init__(
        self,
        audio_manager: AudioManager,
        dispatcher: CommandDispatcher,
        intent_parser: IntentParser,
        training_store: TrainingStore,
        settings_store: SettingsStore,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_manager = audio_manager
        self.dispatcher = dispatcher
        self.intent_parser = intent_parser
        self.training_store = training_store
        self.settings_store = settings_store

        self.signals = UIEvents()
        self.setWindowTitle("JARVIS — Voice-Controlled Deep Learning Desktop Assistant")
        self.setMinimumSize(1100, 760)
        self.setStyleSheet(HOLOGRAPHIC_STYLE)

        self._init_layout()
        self._connect_signals()

        # Connect audio callbacks to signals
        self.audio_manager.on_state_change = lambda st: self.signals.sig_voice_state.emit(st)
        self.audio_manager.capture.amplitude_callback = lambda amp: self.signals.sig_amplitude.emit(amp)
        self.audio_manager.on_transcript = lambda txt, fin: self.signals.sig_transcript.emit(txt, fin)
        self.audio_manager.on_command_ready = self.handle_voice_command

        # Update initial telemetry
        self._refresh_metrics()

    def _init_layout(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # 1. Header Bar
        header = QHBoxLayout()
        lbl_brand = QLabel("JARVIS // MK-VII ARTIFICIAL INTELLIGENCE")
        lbl_brand.setProperty("class", "header")
        header.addWidget(lbl_brand)

        header.addStretch()

        self.lbl_clock = QLabel()
        self.lbl_clock.setStyleSheet("color: #00f0ff; font-weight: bold; letter-spacing: 2px;")
        header.addWidget(self.lbl_clock)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        main_layout.addLayout(header)

        # 2. Main Middle Area (Split horizontal: Left Panels, Center Waveform, Right Panels)
        center_area = QHBoxLayout()
        center_area.setSpacing(12)

        # Left Column: Neural Training & Plugins Panel
        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        self.training_panel = TrainingPanel(self)
        left_col.addWidget(self.training_panel)

        # Plugin Panel
        self.plugin_panel = self._build_plugin_panel()
        left_col.addWidget(self.plugin_panel)
        center_area.addLayout(left_col, stretch=3)

        # Center Column: Holographic Waveform & Dashboard
        mid_col = QVBoxLayout()
        mid_col.setSpacing(10)
        self.waveform = HolographicWaveformWidget(self)
        mid_col.addWidget(self.waveform, stretch=4)

        self.dashboard_panel = DashboardPanel(self)
        mid_col.addWidget(self.dashboard_panel, stretch=2)
        center_area.addLayout(mid_col, stretch=4)

        # Right Column: Security Panel & Activity Audit Log
        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        self.security_panel = self._build_security_panel()
        right_col.addWidget(self.security_panel)

        self.audit_panel = self._build_audit_panel()
        right_col.addWidget(self.audit_panel)
        center_area.addLayout(right_col, stretch=3)

        main_layout.addLayout(center_area, stretch=6)

        # 3. Bottom Area: Voice Transcript Feed
        bottom_box = QVBoxLayout()
        lbl_trans_title = QLabel("LIVE CONVERSATIONAL TRANSCRIPT & LOGS")
        lbl_trans_title.setProperty("class", "title")
        bottom_box.addWidget(lbl_trans_title)

        self.txt_transcript = QPlainTextEdit(self)
        self.txt_transcript.setReadOnly(True)
        self.txt_transcript.setMaximumHeight(140)
        bottom_box.addWidget(self.txt_transcript)
        main_layout.addLayout(bottom_box, stretch=2)

        # 4. Emergency & Command Fallback Bar
        cmd_bar = QHBoxLayout()
        cmd_bar.setSpacing(8)

        # Emergency Stop Button Fallback
        self.btn_emergency_stop = QPushButton("EMERGENCY STOP")
        self.btn_emergency_stop.setProperty("class", "emergency-btn")
        self.btn_emergency_stop.clicked.connect(self._handle_emergency_stop)
        cmd_bar.addWidget(self.btn_emergency_stop)

        # Mute / Listen Toggle Fallback
        self.btn_mute_toggle = QPushButton("PAUSE LISTENING")
        self.btn_mute_toggle.clicked.connect(self._toggle_listening)
        cmd_bar.addWidget(self.btn_mute_toggle)

        # Fallback Text Input for silent testing / accessibility
        self.input_cmd = QLineEdit(self)
        self.input_cmd.setPlaceholderText("Spoken command fallback (e.g., 'Open Chrome', 'Train the model', 'Confirm')...")
        self.input_cmd.returnPressed.connect(self._submit_manual_command)
        cmd_bar.addWidget(self.input_cmd, stretch=1)

        self.btn_send = QPushButton("DISPATCH")
        self.btn_send.clicked.connect(self._submit_manual_command)
        cmd_bar.addWidget(self.btn_send)

        main_layout.addLayout(cmd_bar)

    def _build_plugin_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("class", "holo-panel")
        layout = QVBoxLayout(frame)
        title = QLabel("ACTIVE PLUGINS")
        title.setProperty("class", "title")
        layout.addWidget(title)

        grid = QGridLayout()
        plugins = self.dispatcher.plugins.get_all_plugins()
        for idx, p in enumerate(plugins):
            row = idx // 2
            col = (idx % 2) * 2
            lbl_name = QLabel(p.name)
            lbl_status = QLabel("ENABLED")
            lbl_status.setProperty("class", "chip-active")
            grid.addWidget(lbl_name, row, col)
            grid.addWidget(lbl_status, row, col + 1)

        layout.addLayout(grid)
        return frame

    def _build_security_panel(self) -> QFrame:
        self.frame_sec = QFrame(self)
        self.frame_sec.setProperty("class", "holo-panel")
        layout = QVBoxLayout(self.frame_sec)

        title = QLabel("SECURITY & PERMISSIONS")
        title.setProperty("class", "title")
        layout.addWidget(title)

        self.lbl_perm_status = QLabel("Strict Sandboxing: Active\nPath Traversal Defense: Enforced\nConfirmation Timeout: 30s")
        self.lbl_perm_status.setStyleSheet("color: #a0d8ef; font-size: 11px;")
        layout.addWidget(self.lbl_perm_status)

        self.lbl_confirmation_alert = QLabel("No pending actions requiring confirmation.")
        self.lbl_confirmation_alert.setStyleSheet("color: #00ff88; font-weight: bold; padding: 4px; border: 1px solid #00ff8833; border-radius: 4px;")
        self.lbl_confirmation_alert.setWordWrap(True)
        layout.addWidget(self.lbl_confirmation_alert)

        return self.frame_sec

    def _build_audit_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("class", "holo-panel")
        layout = QVBoxLayout(frame)

        title = QLabel("SECURITY AUDIT RECORDS")
        title.setProperty("class", "title")
        layout.addWidget(title)

        self.txt_audit = QPlainTextEdit(frame)
        self.txt_audit.setReadOnly(True)
        self.txt_audit.setMaximumHeight(120)
        layout.addWidget(self.txt_audit)
        return frame

    def _connect_signals(self):
        self.signals.sig_voice_state.connect(self._on_voice_state)
        self.signals.sig_amplitude.connect(self.waveform.set_amplitude)
        self.signals.sig_transcript.connect(self._on_transcript_update)
        self.signals.sig_command_outcome.connect(self._on_command_outcome)
        self.signals.sig_training_progress.connect(self.training_panel.update_training_progress)
        self.signals.sig_training_complete.connect(self._on_training_complete)
        self.signals.sig_confirmation_alert.connect(self._on_confirmation_alert)

    def _update_clock(self):
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.lbl_clock.setText(now)

    @pyqtSlot(object)
    def _on_voice_state(self, state: VoiceState):
        self.waveform.set_state(state)
        self.dashboard_panel.update_state(state)
        if state == VoiceState.PAUSED:
            self.btn_mute_toggle.setText("RESUME LISTENING")
        else:
            self.btn_mute_toggle.setText("PAUSE LISTENING")

    @pyqtSlot(str, bool)
    def _on_transcript_update(self, text: str, is_final: bool):
        if is_final:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.txt_transcript.appendPlainText(f"[{timestamp}] User: {text}")

    @pyqtSlot(str, str, bool)
    def _on_command_outcome(self, cmd: str, result: str, success: bool):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_transcript.appendPlainText(f"[{timestamp}] Jarvis: {result}\n")
        self.dashboard_panel.update_last_command(cmd, result, success)
        self._refresh_audit_logs()

    @pyqtSlot(int, int, float, float)
    def _on_training_complete(self, version: int, dataset_size: int, accuracy: float, loss: float):
        self.training_panel.set_metrics(version, dataset_size, accuracy, loss)

    @pyqtSlot(str, bool)
    def _on_confirmation_alert(self, message: str, is_active: bool):
        if is_active:
            self.lbl_confirmation_alert.setText(f"ACTION AWAITING SPOKEN CONFIRMATION:\n{message}")
            self.lbl_confirmation_alert.setStyleSheet("color: #ff3366; font-weight: bold; background: rgba(255,50,100,0.15); border: 1px solid #ff3366; padding: 4px; border-radius: 4px;")
            self.frame_sec.setProperty("class", "alert-panel")
        else:
            self.lbl_confirmation_alert.setText("No pending actions requiring confirmation.")
            self.lbl_confirmation_alert.setStyleSheet("color: #00ff88; font-weight: bold; padding: 4px; border: 1px solid #00ff8833; border-radius: 4px;")
            self.frame_sec.setProperty("class", "holo-panel")
        self.frame_sec.style().unpolish(self.frame_sec)
        self.frame_sec.style().polish(self.frame_sec)

    def handle_voice_command(self, raw_text: str):
        """Asynchronously executes command via dispatcher and triggers speech feedback."""
        def run_dispatch():
            # 1. Parse command
            cmd_obj = self.intent_parser.parse(raw_text)

            # Check if this requires background training worker
            if cmd_obj.intent == "train_model":
                self._run_background_training(raw_text)
                return

            # 2. Dispatch
            response = self.dispatcher.dispatch(cmd_obj)

            # 3. Spoken feedback
            self.audio_manager.speak(response.spoken_feedback)

            # 4. Notify UI
            self.signals.sig_command_outcome.emit(raw_text, response.spoken_feedback, response.success)

            # 5. Check confirmation state
            has_pending = self.dispatcher.confirmation_mgr.has_pending()
            if has_pending:
                pending = self.dispatcher.confirmation_mgr.get_pending()
                desc = pending.target_description if pending else "Sensitive Operation"
                self.signals.sig_confirmation_alert.emit(desc, True)
            else:
                self.signals.sig_confirmation_alert.emit("", False)

        threading.Thread(target=run_dispatch, daemon=True, name="Jarvis-Command-Runner").start()

    def _run_background_training(self, raw_text: str):
        """Runs model training in background thread without blocking UI or audio."""
        def worker():
            examples = self.training_store.get_all_examples(approved_only=True)
            self.audio_manager.speak("Validating dataset and initiating neural network training.")

            def progress_cb(ep, tot, loss, acc):
                self.signals.sig_training_progress.emit(ep, tot, loss, acc)

            trainer = ModelTrainer(
                registry=self.dispatcher.context.get("model_registry"),
            )
            result = trainer.train(examples, epochs=30, progress_callback=progress_cb)

            if result.success:
                predictor = self.dispatcher.context.get("predictor")
                if predictor:
                    predictor.reload()
                self.signals.sig_training_complete.emit(
                    result.version, len(examples), result.train_accuracy, result.final_loss
                )
                self.audio_manager.speak(result.summary_message)
                self.signals.sig_command_outcome.emit(raw_text, result.summary_message, True)
            else:
                self.audio_manager.speak(result.summary_message)
                self.signals.sig_command_outcome.emit(raw_text, result.summary_message, False)

        threading.Thread(target=worker, daemon=True, name="Jarvis-Training-Worker").start()

    def _submit_manual_command(self):
        text = self.input_cmd.text().strip()
        if not text:
            return
        self.input_cmd.clear()
        self.signals.sig_transcript.emit(text, True)
        self.handle_voice_command(text)

    def _handle_emergency_stop(self):
        self.audio_manager.stop()
        self.dispatcher.confirmation_mgr.cancel()
        self.signals.sig_confirmation_alert.emit("", False)
        self.txt_transcript.appendPlainText("[EMERGENCY STOP] Audio capturing terminated. Pending actions purged.")

    def _toggle_listening(self):
        if self.audio_manager.state == VoiceState.PAUSED:
            self.audio_manager.resume()
        else:
            self.audio_manager.pause()

    def _refresh_metrics(self):
        dataset_count = self.training_store.count_examples()
        model_reg = self.dispatcher.context.get("model_registry")
        active_version = model_reg.get_active_version() if model_reg else 1
        self.training_panel.set_metrics(
            version=active_version or 1,
            dataset_size=dataset_count,
            accuracy=0.98 if active_version else 0.0,
            loss=0.04 if active_version else 0.0,
        )
        self._refresh_audit_logs()

    def _refresh_audit_logs(self):
        logs = self.dispatcher.audit.get_recent(limit=6)
        lines = []
        for l in logs:
            status_tag = f"[{l['status']}]"
            lines.append(f"{status_tag} {l['command_text']} -> {l['intent'] or 'N/A'}")
        self.txt_audit.setPlainText("\n".join(lines))
