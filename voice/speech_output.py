"""Offline Text-To-Speech (TTS) Engine for Jarvis.
Provides asynchronous speech synthesis with loopback feedback suppression.
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional
import pyttsx3

logger = logging.getLogger(__name__)


class SpeechOutputEngine:
    """Manages pyttsx3 text-to-speech output in a dedicated background worker thread."""

    def __init__(
        self,
        rate: int = 180,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
        on_speaking_state_changed: Optional[Callable[[bool], None]] = None,
    ):
        self.rate = rate
        self.volume = volume
        self.voice_id = voice_id
        self.on_speaking_state_changed = on_speaking_state_changed

        self._queue: queue.Queue = queue.Queue()
        self._is_speaking = False
        self._is_running = True
        self._lock = threading.Lock()

        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True, name="Jarvis-TTS-Worker")
        self._worker_thread.start()

    @property
    def is_speaking(self) -> bool:
        """Indicates if Jarvis is currently outputting synthesized speech."""
        with self._lock:
            return self._is_speaking

    def _set_speaking(self, speaking: bool):
        with self._lock:
            self._is_speaking = speaking
        if self.on_speaking_state_changed:
            try:
                self.on_speaking_state_changed(speaking)
            except Exception as e:
                logger.error(f"Error in speaking state callback: {e}")

    def speak(self, text: str, block: bool = False):
        """Queues a phrase for spoken synthesis."""
        cleaned = text.strip()
        if not cleaned:
            return

        if block:
            event = threading.Event()
            self._queue.put((cleaned, event))
            event.wait(timeout=10.0)
        else:
            self._queue.put((cleaned, None))

    def stop_speaking(self):
        """Clears pending speech queue and stops active speech."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._set_speaking(False)

    def _speech_worker(self):
        """Dedicated background thread initializing pyttsx3 and processing queue."""
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3 engine: {e}")

        while self._is_running:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            text, done_event = item
            logger.info(f"Jarvis Speaking: \"{text}\"")

            self._set_speaking(True)

            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    logger.error(f"Error during pyttsx3 utterance: {e}")
                    # Reinitialize if crashed
                    try:
                        engine = pyttsx3.init()
                    except Exception:
                        pass
            else:
                # Simulated delay if pyttsx3 unavailable
                time.sleep(min(2.0, len(text) * 0.05))

            self._set_speaking(False)
            if done_event is not None:
                done_event.set()

    def shutdown(self):
        """Stops worker thread."""
        self._is_running = False
        self.stop_speaking()
