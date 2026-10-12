"""Audio Manager and State Machine for Jarvis.
Coordinates microphone capture, wake-word spotting, offline transcription, and speech output.
"""

import enum
import logging
import threading
import time
from typing import Callable, Optional

from .audio_capture import AudioCaptureEngine
from .speech_output import SpeechOutputEngine
from .speech_recognizer import OfflineSpeechRecognizer
from .wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


class VoiceState(str, enum.Enum):
    IDLE = "IDLE"
    LISTENING_FOR_WAKE = "LISTENING_FOR_WAKE"
    LISTENING_FOR_COMMAND = "LISTENING_FOR_COMMAND"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    PAUSED = "PAUSED"
    MUTED = "MUTED"
    ERROR = "ERROR"


class AudioManager:
    """Orchestrates the continuous voice processing loop and state machine."""

    def __init__(
        self,
        audio_capture: Optional[AudioCaptureEngine] = None,
        wake_detector: Optional[WakeWordDetector] = None,
        speech_recognizer: Optional[OfflineSpeechRecognizer] = None,
        speech_output: Optional[SpeechOutputEngine] = None,
        on_state_change: Optional[Callable[[VoiceState], None]] = None,
        on_transcript: Optional[Callable[[str, bool], None]] = None,
        on_command_ready: Optional[Callable[[str], None]] = None,
        command_timeout_seconds: float = 6.0,
    ):
        self.capture = audio_capture or AudioCaptureEngine()
        self.wake_detector = wake_detector or WakeWordDetector()
        self.recognizer = speech_recognizer or OfflineSpeechRecognizer()
        self.speech_output = speech_output or SpeechOutputEngine(on_speaking_state_changed=self._handle_speaking_state)
        # An externally built TTS engine has no state callback; hook ours in so state/loopback logic works.
        if speech_output is not None and getattr(speech_output, "on_speaking_state_changed", None) is None:
            speech_output.on_speaking_state_changed = self._handle_speaking_state

        self.on_state_change = on_state_change
        self.on_transcript = on_transcript
        self.on_command_ready = on_command_ready
        self.command_timeout_seconds = command_timeout_seconds

        self._state = VoiceState.IDLE
        self._state_before_speaking = VoiceState.IDLE
        self._is_running = False
        self._paused = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._command_start_time: float = 0.0
        self._accumulated_command: str = ""

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def voice_ready(self) -> bool:
        """True when speech-to-text is actually available (model loaded)."""
        return bool(getattr(self.recognizer, "is_ready", False))

    @property
    def voice_status(self) -> str:
        """Explains why voice recognition is unavailable, or '' when it is working."""
        return "" if self.voice_ready else getattr(self.recognizer, "status_message", "") or "Speech recognizer not ready."

    def set_state(self, new_state: VoiceState):
        if self._state != new_state:
            self._state = new_state
            logger.info(f"Voice state transitioned to: {new_state.value}")
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")

    def _handle_speaking_state(self, is_speaking: bool):
        """Suppresses microphone processing when Jarvis is talking to prevent loopback."""
        if is_speaking:
            if self._state != VoiceState.SPEAKING:
                self._state_before_speaking = self._state
            self.set_state(VoiceState.SPEAKING)
        else:
            if self._paused:
                self.set_state(VoiceState.PAUSED)
            elif not self._is_running:
                self.set_state(VoiceState.IDLE)
            elif self._state_before_speaking == VoiceState.LISTENING_FOR_COMMAND:
                # "At your service" was spoken; keep waiting for the actual command.
                self.set_state(VoiceState.LISTENING_FOR_COMMAND)
            else:
                self.set_state(VoiceState.LISTENING_FOR_WAKE)

    def start(self) -> bool:
        """Starts audio capture and initiates background processing thread."""
        if self._is_running:
            return True

        if not self.voice_ready:
            logger.error(f"Voice recognition is unavailable: {self.voice_status}")

        if not self.capture.start():
            self.set_state(VoiceState.ERROR)
            return False

        self._is_running = True
        self._paused = False
        self.set_state(VoiceState.LISTENING_FOR_WAKE)

        self._worker_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="Jarvis-Audio-Loop"
        )
        self._worker_thread.start()
        return True

    def stop(self):
        """Stops audio capture and halts processing."""
        self._is_running = False
        self.capture.stop()
        self.set_state(VoiceState.IDLE)

    def pause(self):
        """Temporarily pauses listening."""
        self._paused = True
        self.set_state(VoiceState.PAUSED)

    def resume(self):
        """Resumes listening."""
        self._paused = False
        self.set_state(VoiceState.LISTENING_FOR_WAKE)

    def select_device(self, device_index: Optional[int]):
        """Switches active microphone device."""
        self.capture.set_device(device_index)

    def speak(self, text: str, block: bool = False):
        """Synthesizes speech via TTS engine."""
        self.speech_output.speak(text, block=block)

    def inject_text_command(self, text: str):
        """Directly injects text as a recognized command (fallback/testing)."""
        if self.on_transcript:
            self.on_transcript(text, True)
        if self.on_command_ready:
            self.on_command_ready(text)

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _safe_call(callback: Optional[Callable], *args):
        """Runs a UI/dispatcher callback without letting its errors kill the audio thread."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            logger.exception("Error in voice callback")

    def _emit_command(self, command_text: str):
        """Hands a recognized command to the application and returns to wake-word listening."""
        self.set_state(VoiceState.PROCESSING)
        self._safe_call(self.on_transcript, command_text, True)
        self._safe_call(self.on_command_ready, command_text)
        self._accumulated_command = ""
        self.recognizer.reset()
        self.set_state(VoiceState.LISTENING_FOR_WAKE)

    def _handle_wake_chunk(self, chunk: bytes):
        is_final, transcript = self.recognizer.process_audio_chunk(chunk)
        if not transcript:
            return

        # Live text for the UI (partials and finals).
        self._safe_call(self.on_transcript, transcript, False)

        # Only act on a *finished* utterance. Acting on partials fired on the bare word "jarvis"
        # before the rest of "jarvis open chrome" was spoken, so the command was thrown away.
        if not is_final:
            return

        is_detected, cmd_remainder = self.wake_detector.detect_in_text(transcript)
        if not is_detected:
            return

        logger.info(f"Wake word detected! Remainder: '{cmd_remainder}'")
        self.recognizer.reset()

        if cmd_remainder:
            # Wake word and command in one breath, e.g. "Jarvis open chrome"
            self._emit_command(cmd_remainder)
        else:
            # Wake word alone: start listening for the command
            self._accumulated_command = ""
            self._command_start_time = time.time()
            self.set_state(VoiceState.LISTENING_FOR_COMMAND)
            self.speech_output.speak("At your service, sir.")

    def _handle_command_chunk(self, chunk: bytes):
        if time.time() - self._command_start_time > self.command_timeout_seconds:
            logger.info("Command listening timed out.")
            self._accumulated_command = ""
            self.recognizer.reset()
            self.set_state(VoiceState.LISTENING_FOR_WAKE)
            return

        is_final, transcript = self.recognizer.process_audio_chunk(chunk)
        if transcript:
            # Still hearing speech, so keep the listening window open.
            self._command_start_time = time.time()
            self._accumulated_command = transcript
            if not is_final:
                self._safe_call(self.on_transcript, transcript, False)

        if is_final and self._accumulated_command:
            self._emit_command(self._accumulated_command)

    def _processing_loop(self):
        """Continuous audio processing loop."""
        was_muted = False
        while self._is_running:
            try:
                # 1. Ignore the mic while paused or while Jarvis is talking (prevents hearing itself).
                if self._paused or self.speech_output.is_speaking:
                    # Discard audio instead of letting it pile up; otherwise Jarvis's own voice
                    # (which contains the wake word) is transcribed the moment it stops talking.
                    self.capture.drain()
                    was_muted = True
                    time.sleep(0.05)
                    continue

                if was_muted:
                    self.capture.drain()
                    self.recognizer.reset()
                    if self._state == VoiceState.LISTENING_FOR_COMMAND:
                        # Give the user the full window *after* the acknowledgement finishes.
                        self._command_start_time = time.time()
                    was_muted = False

                chunk = self.capture.get_chunk(timeout=0.1)
                if chunk is None:
                    continue

                # 2. State-dependent processing
                if self._state == VoiceState.LISTENING_FOR_COMMAND:
                    self._handle_command_chunk(chunk)
                else:
                    self._handle_wake_chunk(chunk)
            except Exception:
                # Never let one bad chunk/callback silently end voice recognition.
                logger.exception("Unexpected error in audio processing loop")
                time.sleep(0.1)
