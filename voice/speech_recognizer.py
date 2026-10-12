"""Offline Speech Recognizer for Jarvis.
Wraps Vosk offline speech recognition engine with error handling and fallback capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

try:
    import vosk
except ImportError:  # Allow the app to start (text-only) when vosk is not installed
    vosk = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _is_vosk_model_dir(path: Path) -> bool:
    """A real Vosk model folder contains at least the `am` and `conf` subfolders."""
    return path.is_dir() and (path / "am").is_dir() and (path / "conf").is_dir()


def _find_model_dir(base: Path, max_depth: int = 2) -> Optional[Path]:
    """Finds a Vosk model folder at `base` or up to `max_depth` levels below it.

    Handles the common mistake of extracting the zip into a doubled folder, e.g.
    models/vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15/.
    """
    if _is_vosk_model_dir(base):
        return base
    if max_depth <= 0 or not base.is_dir():
        return None
    try:
        children = sorted(c for c in base.iterdir() if c.is_dir())
    except OSError:
        return None
    for child in children:
        found = _find_model_dir(child, max_depth - 1)
        if found:
            return found
    return None


def resolve_model_path(model_path: Optional[str], search_models_dir: bool = True) -> Optional[Path]:
    """Locates the Vosk model folder.

    Relative paths are tried against the current working directory *and* the project root, so the
    result no longer depends on where Jarvis was launched from. If the configured path is missing and
    `search_models_dir` is True, any `vosk-model*` folder under <project>/models is used instead.
    """
    candidates = []
    if model_path:
        p = Path(model_path)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([Path.cwd() / p, PROJECT_ROOT / p])

    for candidate in candidates:
        found = _find_model_dir(candidate)
        if found:
            return found

    if search_models_dir:
        models_dir = PROJECT_ROOT / "models"
        if models_dir.is_dir():
            for child in sorted(models_dir.glob("vosk-model*")):
                found = _find_model_dir(child)
                if found:
                    logger.warning(f"Configured Vosk model path not found; using discovered model at {found}")
                    return found
    return None


class OfflineSpeechRecognizer:
    """Performs offline speech-to-text using Vosk Kaldi speech recognition."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        search_models_dir: bool = True,
    ):
        self.sample_rate = sample_rate
        self.requested_path = Path(model_path) if model_path else None
        self.model_path: Optional[Path] = self.requested_path
        self.vosk_model = None
        self.recognizer = None
        self.is_ready = False
        # Human-readable reason shown to the user when speech recognition is unavailable
        self.status_message = ""

        self._init_vosk(search_models_dir)

    def _init_vosk(self, search_models_dir: bool = True):
        """Attempts to initialize the Vosk offline model."""
        if vosk is None:
            self.status_message = "The 'vosk' package is not installed. Run: pip install vosk"
            logger.error(self.status_message)
            return

        resolved = resolve_model_path(
            str(self.requested_path) if self.requested_path else None,
            search_models_dir=search_models_dir,
        )
        if resolved is None:
            self.status_message = (
                f"Vosk speech model not found (looked for '{self.requested_path}'). Voice commands are DISABLED. "
                "Download vosk-model-small-en-us-0.15 from https://alphacephei.com/vosk/models and extract it to "
                f"{PROJECT_ROOT / 'models' / 'vosk-model-small-en-us-0.15'}"
            )
            logger.error(self.status_message)
            return

        self.model_path = resolved
        try:
            # Silence internal Kaldi logs
            vosk.SetLogLevel(-1)
            self.vosk_model = vosk.Model(str(resolved))
            self.recognizer = vosk.KaldiRecognizer(self.vosk_model, float(self.sample_rate))
            self.is_ready = True
            logger.info(f"Vosk offline model successfully loaded from {resolved}")
        except Exception as e:
            self.status_message = f"Failed to initialize Vosk model at {resolved}: {e}"
            logger.error(self.status_message)
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
