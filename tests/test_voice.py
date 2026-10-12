"""Tests for Voice Subsystem: Wake Word, Recognition, TTS Loopback Prevention, and Audio State.
"""

from unittest.mock import MagicMock
import pytest

from voice.audio_capture import AudioCaptureEngine
from voice.audio_manager import AudioManager, VoiceState
from voice.speech_output import SpeechOutputEngine
from voice.speech_recognizer import OfflineSpeechRecognizer, resolve_model_path
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
    recognizer = OfflineSpeechRecognizer(model_path="non_existent_model_dir_xyz", search_models_dir=False)
    assert recognizer.is_ready is False
    assert "not found" in recognizer.status_message or "vosk" in recognizer.status_message
    # Should not throw exception when processing empty chunk
    is_final, text = recognizer.process_audio_chunk(b"0" * 3200)
    assert is_final is False
    assert text == ""


def _make_fake_model(path):
    (path / "am").mkdir(parents=True)
    (path / "conf").mkdir()
    return path


def test_resolve_model_path_finds_nested_and_relative(tmp_path, monkeypatch):
    """The Vosk model is found even when unzipped into a doubled folder or launched from another cwd."""
    nested = _make_fake_model(tmp_path / "models" / "vosk-model-small-en-us-0.15" / "vosk-model-small-en-us-0.15")

    # Absolute path to the outer folder resolves to the inner real model
    outer = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    assert resolve_model_path(str(outer), search_models_dir=False) == nested

    # Relative path resolves against the current directory
    monkeypatch.chdir(tmp_path)
    assert resolve_model_path("models/vosk-model-small-en-us-0.15", search_models_dir=False) == nested

    # Missing path with discovery disabled returns None
    assert resolve_model_path("nope", search_models_dir=False) is None


def test_wake_word_tolerates_misheard_start_only():
    detector = WakeWordDetector(wake_word="jarvis", aliases=["jarvis", "hey jarvis"])

    detected, remainder = detector.detect_in_text("jervis open chrome")
    assert detected is True
    assert remainder == "open chrome"

    # Similar-looking words must not trigger, and neither must ordinary commands
    for text in ["harvest the data", "open chrome", "what is the service status"]:
        detected, _ = detector.detect_in_text(text)
        assert detected is False


class _FakeCapture:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.drained = 0

    def get_chunk(self, timeout=0.5):
        return self.chunks.pop(0) if self.chunks else None

    def drain(self):
        self.drained += 1


class _FakeRecognizer:
    """Returns scripted (is_final, text) results, one per audio chunk."""

    is_ready = True
    status_message = ""

    def __init__(self, script):
        self.script = list(script)
        self.resets = 0

    def process_audio_chunk(self, data):
        return self.script.pop(0) if self.script else (False, "")

    def reset(self):
        self.resets += 1


class _FakeTTS:
    def __init__(self):
        self.is_speaking = False
        self.on_speaking_state_changed = None
        self.spoken = []

    def speak(self, text, block=False):
        self.spoken.append(text)


def _manager(script, commands, **kwargs):
    manager = AudioManager(
        audio_capture=_FakeCapture([b"x"] * len(script)),
        wake_detector=WakeWordDetector(wake_word="jarvis", aliases=["jarvis"]),
        speech_recognizer=_FakeRecognizer(script),
        speech_output=_FakeTTS(),
        on_command_ready=commands.append,
        **kwargs,
    )
    manager.set_state(VoiceState.LISTENING_FOR_WAKE)
    return manager


def test_wake_word_and_command_in_one_utterance_is_not_lost():
    """Partial 'jarvis' must not hijack the utterance; the full 'jarvis open chrome' becomes a command."""
    commands = []
    manager = _manager([(False, "jarvis"), (False, "jarvis open"), (True, "jarvis open chrome")], commands)

    for _ in range(3):
        manager._handle_wake_chunk(b"x")

    assert commands == ["open chrome"]
    assert manager.speech_output.spoken == []  # no stray "At your service" mid-sentence
    assert manager.state == VoiceState.LISTENING_FOR_WAKE


def test_wake_word_alone_then_separate_command():
    commands = []
    manager = _manager([(True, "jarvis"), (False, "what is"), (True, "what is the cpu usage")], commands)

    manager._handle_wake_chunk(b"x")
    assert manager.state == VoiceState.LISTENING_FOR_COMMAND
    assert manager.speech_output.spoken == ["At your service, sir."]

    manager._handle_command_chunk(b"x")
    manager._handle_command_chunk(b"x")
    assert commands == ["what is the cpu usage"]
    assert manager.state == VoiceState.LISTENING_FOR_WAKE


def test_command_listening_times_out_back_to_wake():
    commands = []
    manager = _manager([(False, "")], commands, command_timeout_seconds=0.0)
    manager.set_state(VoiceState.LISTENING_FOR_COMMAND)
    manager._command_start_time = 0.0

    manager._handle_command_chunk(b"x")
    assert manager.state == VoiceState.LISTENING_FOR_WAKE
    assert commands == []


def test_speaking_state_restores_command_listening():
    """After 'At your service' finishes, the manager must keep listening for the command."""
    manager = _manager([], [])
    manager._is_running = True
    manager.set_state(VoiceState.LISTENING_FOR_COMMAND)

    manager._handle_speaking_state(True)
    assert manager.state == VoiceState.SPEAKING
    manager._handle_speaking_state(False)
    assert manager.state == VoiceState.LISTENING_FOR_COMMAND


def test_external_tts_gets_state_callback_wired():
    manager = _manager([], [])
    assert manager.speech_output.on_speaking_state_changed == manager._handle_speaking_state


def test_loop_survives_callback_errors_and_drains_while_speaking():
    """A crashing command handler must not kill the audio thread; echo audio is discarded while speaking."""
    import threading
    import time

    calls = []

    def bad_handler(text):
        calls.append(text)
        raise RuntimeError("dispatcher blew up")

    manager = _manager([(True, "jarvis open chrome"), (True, "jarvis open notepad")], [])
    manager.on_command_ready = bad_handler
    manager._is_running = True

    manager.speech_output.is_speaking = True  # Jarvis is talking first
    thread = threading.Thread(target=manager._processing_loop, daemon=True)
    thread.start()
    time.sleep(0.2)
    assert calls == []  # nothing processed while speaking
    assert manager.capture.drained > 0

    manager.speech_output.is_speaking = False
    time.sleep(0.5)
    manager._is_running = False
    thread.join(timeout=2)

    assert calls == ["open chrome", "open notepad"]  # second command still handled after first raised
