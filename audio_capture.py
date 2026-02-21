"""
Audio Capture Subsystem
Captures system audio using loopback devices.
"""

import threading
from collections import deque
import soundcard as sc

class AudioCaptureSubsystem:
    """
    Captures system audio in a background thread and buffers it for consumption.
    Uses a deque to store only the latest audio chunk, discarding older data to minimize latency.
    """
    def __init__(self, sample_rate=44100, buffer_frames=1024):
        self.sample_rate = sample_rate
        self.buffer_frames = buffer_frames
        self.audio_queue = deque(maxlen=1)
        self.is_running = False
        self.capture_thread = None

    def start(self):
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()

    def _capture_loop(self):
        try:
            # Using default speaker in loopback mode
            default_speaker = sc.default_speaker()
            mics = sc.all_microphones(include_loopback=True)

            # Find the loopback microphone for the default speaker
            loopback_mic = None
            for mic in mics:
                if mic.isloopback and default_speaker.id in mic.id:
                    loopback_mic = mic
                    break

            if not loopback_mic:
                # Fallback to the default loopback if specific one not found
                for mic in mics:
                    if mic.isloopback and default_speaker.name in mic.name:
                        loopback_mic = mic
                        break

            if not loopback_mic:
                # Absolute fallback: just get the first loopback
                for mic in mics:
                    if mic.isloopback:
                        loopback_mic = mic
                        break

            print(f"Using default speaker: {default_speaker.name}")
            print(f"Selected loopback mic: {loopback_mic.name if loopback_mic else 'None'}")

            with loopback_mic.recorder(samplerate=self.sample_rate) as mic:
                while self.is_running:
                    # Record a chunk of audio
                    data = mic.record(numframes=self.buffer_frames)

                    # Convert to mono by averaging channels
                    mono_data = data.mean(axis=1)

                    # Put in deque, automatically discards oldest if full (maxlen=1)
                    self.audio_queue.append(mono_data)
        except Exception as e:
            print(f"Error in audio capture: {e}")

    def get_latest_data(self):
        """Retrieve the most recent audio chunk, if available."""
        try:
            return self.audio_queue.pop()
        except IndexError:
            return None
