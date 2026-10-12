"""Jarvis User Interface Subsystem.
"""

from .dashboard import DashboardPanel
from .main_window import JarvisMainWindow
from .styles import HOLOGRAPHIC_STYLE
from .training_panel import TrainingPanel
from .waveform import HolographicWaveformWidget

__all__ = [
    "HolographicWaveformWidget",
    "DashboardPanel",
    "TrainingPanel",
    "JarvisMainWindow",
    "HOLOGRAPHIC_STYLE",
]
