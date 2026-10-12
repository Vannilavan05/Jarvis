"""Tests for Voice Subsystem: Wake Word, Recognition, TTS Loopback Prevention, and Audio State.
"""

from unittest.mock import MagicMock
import pytest

from voice.audio_capture import AudioCaptureEngine
from voice.audio_manager import AudioManager, VoiceState
from voice.speech_output import SpeechOutputEngine
from voice.speech_recognizer import OfflineSpeechRecognizer
from voice.wake_word import WakeWordDetector


def test_wake_word_detection():
    detector = WakeWordDetector(wake_word="jarvis", aliases=["jarvis", "hey jarvis"])

    detected, remainder = detector.detect_in_text("jarvis open chrome")
    assert detected is True
    assert remainder == "open chrome"

    detected, remainder = detector.detect_in_text("hey jarvis, what is the cpu usage")
    assert detected is True
    assert remainder == "what is the cpu usage"

    detected, remainder = detector.detect_in_text("open notepad")
    assert detected is False
    assert remainder == ""


def test_tts_loopback_prevention():
    """Verifies that while TTS is outputting speech, microphone processing is suppressed."""
    mock_capture = MagicMock(spec=AudioCaptureEngine)
    mock_wake = MagicMock(spec=WakeWordDetector)
    mock_recognizer = MagicMock(spec=OfflineSpeechRecognizer)
    mock_tts = MagicMock(spec=SpeechOutputEngine)

    manager = AudioManager(
        audio_capture=mock_capture,
        wake_detector=mock_wake,
        speech_recognizer=mock_recognizer,
        speech_output=mock_tts,
    )

    # Simulate TTS starts speaking
    mock_tts.is_speaking = True
    manager._handle_speaking_state(True)
    assert manager.state == VoiceState.SPEAKING

    # Simulate TTS finishes speaking
    mock_tts.is_speaking = False
    manager._is_running = True
    manager._handle_speaking_state(False)
    assert manager.state == VoiceState.LISTENING_FOR_WAKE


def test_audio_device_enumeration():
    devices = AudioCaptureEngine.get_input_devices()
    assert isinstance(devices, list)


def test_offline_recognizer_fallback():
    """Checks that speech recognizer initializes safely when model path does not exist."""
    recognizer = OfflineSpeechRecognizer(model_path="non_existent_model_dir_xyz")
    assert recognizer.is_ready is False
    # Should not throw exception when processing empty chunk
    is_final, text = recognizer.process_audio_chunk(b"0" * 3200)
    assert is_final is False
    assert text == ""
