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

        self.on_state_change = on_state_change
        self.on_transcript = on_transcript
        self.on_command_ready = on_command_ready
        self.command_timeout_seconds = command_timeout_seconds

        self._state = VoiceState.IDLE
        self._is_running = False
        self._paused = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._command_start_time: float = 0.0
        self._accumulated_command: str = ""

    @property
    def state(self) -> VoiceState:
        return self._state

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
            self.set_state(VoiceState.SPEAKING)
        else:
            if not self._paused and self._is_running:
                self.set_state(VoiceState.LISTENING_FOR_WAKE)
            else:
                self.set_state(VoiceState.PAUSED if self._paused else VoiceState.IDLE)

    def start(self) -> bool:
        """Starts audio capture and initiates background processing thread."""
        if self._is_running:
            return True

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

    def _processing_loop(self):
        """Continuous audio processing loop."""
        while self._is_running:
            # 1. Check if paused or muted due to Jarvis speaking
            if self._paused or self.speech_output.is_speaking:
                time.sleep(0.05)
                continue

            chunk = self.capture.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            # 2. State-dependent processing
            if self._state == VoiceState.LISTENING_FOR_WAKE:
                is_final, transcript = self.recognizer.process_audio_chunk(chunk)
                text_to_check = transcript if is_final else transcript

                if text_to_check:
                    if self.on_transcript:
                        self.on_transcript(text_to_check, False)

                    is_detected, cmd_remainder = self.wake_detector.detect_in_text(text_to_check)
                    if is_detected:
                        logger.info(f"Wake word detected! Remainder: '{cmd_remainder}'")
                        self.recognizer.reset()

                        if cmd_remainder:
                            # User said wake word and command together, e.g. "Jarvis open chrome"
                            self.set_state(VoiceState.PROCESSING)
                            if self.on_transcript:
                                self.on_transcript(cmd_remainder, True)
                            if self.on_command_ready:
                                self.on_command_ready(cmd_remainder)
                            self.set_state(VoiceState.LISTENING_FOR_WAKE)
                        else:
                            # Wake word alone spoken, activate command listening
                            self.set_state(VoiceState.LISTENING_FOR_COMMAND)
                            self._command_start_time = time.time()
                            self._accumulated_command = ""
                            self.speech_output.speak("At your service, sir.")

            elif self._state == VoiceState.LISTENING_FOR_COMMAND:
                # Check for command timeout
                if time.time() - self._command_start_time > self.command_timeout_seconds:
                    logger.info("Command listening timed out.")
                    self.set_state(VoiceState.LISTENING_FOR_WAKE)
                    continue

                is_final, transcript = self.recognizer.process_audio_chunk(chunk)
                if transcript:
                    if self.on_transcript:
                        self.on_transcript(transcript, False)
                    self._accumulated_command = transcript

                if is_final and self._accumulated_command:
                    final_text = self._accumulated_command
                    self.set_state(VoiceState.PROCESSING)
                    if self.on_transcript:
                        self.on_transcript(final_text, True)
                    if self.on_command_ready:
                        self.on_command_ready(final_text)
                    self.recognizer.reset()
                    self.set_state(VoiceState.LISTENING_FOR_WAKE)
