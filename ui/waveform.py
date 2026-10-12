"""Holographic Arc Reactor Waveform Widget for Jarvis.
Renders concentric rotating rings, pulsing energy cores, and audio-reactive radial spikes using QPainter.
"""

import math
from PyQt6.QtCore import QPointF, QRectF, QTimer, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QWidget

try:
    from voice.audio_manager import VoiceState
except (ImportError, ValueError):
    from jarvis.voice.audio_manager import VoiceState


class HolographicWaveformWidget(QWidget):
    """Animated Arc Reactor waveform displaying real-time audio reactivity and assistant states."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)

        self.angle_outer = 0.0
        self.angle_inner = 0.0
        self.pulse_phase = 0.0
        self.current_amplitude = 0.0
        self.smoothed_amplitude = 0.0
        self.state = VoiceState.IDLE

        # Animation timer at ~30 FPS
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._update_animation)
        self.anim_timer.start(33)

    def set_amplitude(self, amplitude: float):
        """Updates instantaneous audio amplitude level (0.0 to 1.0)."""
        self.current_amplitude = max(0.0, min(1.0, amplitude))

    def set_state(self, state: VoiceState):
        """Updates current voice state to alter visual scheme."""
        self.state = state

    def _update_animation(self):
        # Smooth amplitude transitions
        self.smoothed_amplitude += (self.current_amplitude - self.smoothed_amplitude) * 0.25
        # Decay current amplitude
        self.current_amplitude *= 0.85

        # Rotate rings
        self.angle_outer = (self.angle_outer + 1.2) % 360.0
        self.angle_inner = (self.angle_inner - 2.0) % 360.0
        self.pulse_phase = (self.pulse_phase + 0.08) % (2.0 * math.pi)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        center = QPointF(width / 2.0, height / 2.0)
        radius = min(width, height) / 2.0 - 15.0

        if radius <= 10:
            return

        # Choose primary color based on VoiceState
        if self.state in {VoiceState.LISTENING_FOR_COMMAND, VoiceState.SPEAKING}:
            base_color = QColor(0, 240, 255)  # Bright Cyan
            core_glow = QColor(0, 255, 180, 180)
        elif self.state == VoiceState.PROCESSING:
            base_color = QColor(255, 180, 0)  # Amber
            core_glow = QColor(255, 210, 0, 180)
        elif self.state in {VoiceState.ERROR, VoiceState.MUTED}:
            base_color = QColor(255, 60, 80)  # Crimson
            core_glow = QColor(255, 80, 100, 160)
        else:
            base_color = QColor(0, 140, 255)  # Electric Deep Blue
            core_glow = QColor(0, 180, 255, 120)

        # 1. Radial Equalizer Spikes reacting to smoothed_amplitude
        num_spikes = 36
        spike_pen = QPen(base_color, 1.8)
        painter.setPen(spike_pen)
        amp_boost = self.smoothed_amplitude * (radius * 0.35)

        for i in range(num_spikes):
            angle_rad = math.radians(i * (360.0 / num_spikes) + self.angle_outer * 0.5)
            r_in = radius * 0.72
            # Add dynamic sinusoidal wobble + audio amplitude
            wobble = math.sin(angle_rad * 3.0 + self.pulse_phase) * 3.0
            r_out = r_in + 8.0 + amp_boost + wobble
            x1 = center.x() + r_in * math.cos(angle_rad)
            y1 = center.y() + r_in * math.sin(angle_rad)
            x2 = center.x() + r_out * math.cos(angle_rad)
            y2 = center.y() + r_out * math.sin(angle_rad)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # 2. Outer Rotating Dashed Ring
        outer_pen = QPen(base_color, 2.0, Qt.PenStyle.DashLine)
        painter.setPen(outer_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.save()
        painter.translate(center)
        painter.rotate(self.angle_outer)
        r_outer = radius * 0.88
        painter.drawEllipse(QRectF(-r_outer, -r_outer, r_outer * 2.0, r_outer * 2.0))
        painter.restore()

        # 3. Inner Reverse Rotating Tech Ring
        inner_pen = QPen(QColor(base_color.red(), base_color.green(), base_color.blue(), 160), 1.5, Qt.PenStyle.DotLine)
        painter.setPen(inner_pen)
        painter.save()
        painter.translate(center)
        painter.rotate(self.angle_inner)
        r_inner = radius * 0.55
        painter.drawEllipse(QRectF(-r_inner, -r_inner, r_inner * 2.0, r_inner * 2.0))
        painter.restore()

        # 4. Central Pulsing Core
        pulse_scale = 1.0 + 0.12 * math.sin(self.pulse_phase) + self.smoothed_amplitude * 0.4
        core_r = (radius * 0.32) * pulse_scale

        radial_grad = QRadialGradient(center, core_r)
        radial_grad.setColorAt(0.0, QColor(255, 255, 255, 230))
        radial_grad.setColorAt(0.4, core_glow)
        radial_grad.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(radial_grad))
        painter.drawEllipse(center, core_r, core_r)

        # 5. Center Arc Reactor Icon / State Text
        painter.setPen(QPen(QColor(255, 255, 255, 220)))
        font = QFont("Consolas", 9, QFont.Weight.Bold)
        painter.setFont(font)
        state_abbr = {
            VoiceState.IDLE: "IDLE",
            VoiceState.LISTENING_FOR_WAKE: "ONLINE",
            VoiceState.LISTENING_FOR_COMMAND: "VOICE",
            VoiceState.PROCESSING: "WORK",
            VoiceState.SPEAKING: "SPEECH",
            VoiceState.PAUSED: "PAUSE",
            VoiceState.MUTED: "MUTE",
            VoiceState.ERROR: "ERROR",
        }.get(self.state, "JARVIS")
        painter.drawText(
            QRectF(center.x() - 40, center.y() - 10, 80, 20),
            Qt.AlignmentFlag.AlignCenter,
            state_abbr,
        )
