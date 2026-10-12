"""Audio Capture Engine for Jarvis.
Captures microphone streams using sounddevice, computes RMS/amplitude levels, and handles device changes.
"""

import logging
import queue
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioCaptureEngine:
    """Manages audio input streaming and device enumeration."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 4000,
        channels: int = 1,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.device_index = device_index

        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.is_running = False
        self.amplitude_callback: Optional[Callable[[float], None]] = None
        self.latest_rms: float = 0.0

    @staticmethod
    def get_input_devices() -> List[Dict[str, Any]]:
        """Returns a list of available microphone input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for idx, dev in enumerate(device_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": dev.get("name", f"Microphone {idx}"),
                        "channels": dev.get("max_input_channels"),
                        "default_samplerate": dev.get("default_samplerate"),
                    })
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
        return devices

    def set_device(self, device_index: Optional[int]):
        """Switches active input device."""
        was_running = self.is_running
        if was_running:
            self.stop()
        self.device_index = device_index
        if was_running:
            self.start()

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback invoked by sounddevice for each audio block."""
        if status:
            logger.debug(f"Audio stream status: {status}")

        if not self.is_running:
            return

        # indata is numpy float32 array in range [-1.0, 1.0]
        # Calculate RMS amplitude for UI waveform animation
        rms = float(np.sqrt(np.mean(indata**2)))
        self.latest_rms = min(1.0, rms * 5.0)  # Boost visually for waveform

        if self.amplitude_callback:
            self.amplitude_callback(self.latest_rms)

        # Convert to 16-bit PCM bytes for speech recognition engines
        pcm_16 = (indata * 32767).astype(np.int16).tobytes()
        self.audio_queue.put(pcm_16)

    def start(self) -> bool:
        """Starts capturing microphone audio."""
        if self.is_running:
            return True

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device_index,
                channels=self.channels,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.is_running = True
            logger.info("Audio capture stream started successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio input stream: {e}")
            self.is_running = False
            return False

    def stop(self):
        """Stops capturing audio and closes stream."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
            self.stream = None
        # Clear residual queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Audio capture stopped.")

    def get_chunk(self, timeout: float = 0.5) -> Optional[bytes]:
        """Retrieves next captured 16-bit PCM chunk from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
