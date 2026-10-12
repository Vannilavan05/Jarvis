"""Offline Speech Recognizer for Jarvis.
Wraps Vosk offline speech recognition engine with error handling and fallback capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import vosk

logger = logging.getLogger(__name__)


class OfflineSpeechRecognizer:
    """Performs offline speech-to-text using Vosk Kaldi speech recognition."""

    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model_path = Path(model_path) if model_path else None
        self.vosk_model: Optional[vosk.Model] = None
        self.recognizer: Optional[vosk.KaldiRecognizer] = None
        self.is_ready = False

        self._init_vosk()

    def _init_vosk(self):
        """Attempts to initialize the Vosk offline model."""
        if self.model_path and self.model_path.exists():
            try:
                # Silence internal Kaldi logs
                vosk.SetLogLevel(-1)
                self.vosk_model = vosk.Model(str(self.model_path))
                self.recognizer = vosk.KaldiRecognizer(self.vosk_model, float(self.sample_rate))
                self.is_ready = True
                logger.info(f"Vosk offline model successfully loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize Vosk model at {self.model_path}: {e}")
                self.is_ready = False
        else:
            logger.info(
                f"Vosk model path '{self.model_path}' not found. "
                "Running in manual/fallback speech recognition mode. "
                "To enable local voice STT, download vosk-model-small-en-us-0.15 and place in models/."
            )
            self.is_ready = False

    def reset(self):
        """Resets recognizer state between utterances."""
        if self.is_ready and self.vosk_model:
            self.recognizer = vosk.KaldiRecognizer(self.vosk_model, float(self.sample_rate))

    def process_audio_chunk(self, data: bytes) -> Tuple[bool, str]:
        """Feeds a 16-bit PCM chunk into Vosk recognizer.

        Returns (is_final_utterance, text).
        """
        if not self.is_ready or self.recognizer is None:
            return False, ""

        try:
            if self.recognizer.AcceptWaveform(data):
                result_json = self.recognizer.Result()
                res = json.loads(result_json)
                final_text = res.get("text", "").strip()
                return True, final_text
            else:
                partial_json = self.recognizer.PartialResult()
                part = json.loads(partial_json)
                partial_text = part.get("partial", "").strip()
                return False, partial_text
        except Exception as e:
            logger.error(f"Error processing audio in recognizer: {e}")
            return False, ""

    def get_final_result(self) -> str:
        """Flushes and retrieves the final transcript after speech stops."""
        if not self.is_ready or self.recognizer is None:
            return ""

        try:
            final_json = self.recognizer.FinalResult()
            res = json.loads(final_json)
            text = res.get("text", "").strip()
            self.reset()
            return text
        except Exception as e:
            logger.error(f"Error getting final result: {e}")
            return ""
