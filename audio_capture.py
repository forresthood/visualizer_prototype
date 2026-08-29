"""
Audio Capture Subsystem
Captures system audio using loopback devices.
"""

import threading
from collections import deque
import soundcard as sc

# How long stop() waits for the capture thread to notice is_running == False.
# The thread can only observe the flag between record() calls, so a stalled or
# removed device must not be able to block shutdown indefinitely.
JOIN_TIMEOUT_SECONDS = 2.0

class AudioCaptureSubsystem:
    """
    Captures system audio in a background thread and buffers it for consumption.
    Uses a deque to store only the latest audio chunk, discarding older data to minimize latency.

    If capture fails (no loopback device, device removed mid-session, backend
    error) the thread stops, `is_running` goes False and `last_error` holds a
    human-readable reason, so callers can tell a dead capture from a silent one.
    """
    def __init__(self, sample_rate=44100, buffer_frames=1024):
        self.sample_rate = sample_rate
        self.buffer_frames = buffer_frames
        self.audio_queue = deque(maxlen=1)
        self.is_running = False
        self.capture_thread = None
        self.last_error = None

    def start(self):
        self.last_error = None
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.is_running = False
        if self.capture_thread:
            # Bounded join: the thread is a daemon, so if it is wedged inside a
            # blocking record() call we let the interpreter reap it at exit
            # rather than hanging the caller forever.
            self.capture_thread.join(timeout=JOIN_TIMEOUT_SECONDS)
            if self.capture_thread.is_alive():
                print("Warning: audio capture thread did not stop within "
                      f"{JOIN_TIMEOUT_SECONDS}s; abandoning it.")
            self.capture_thread = None

    def is_active(self):
        """True while the capture thread is running and has not failed."""
        return self.is_running and self.last_error is None

    def _fail(self, message):
        """Record a capture failure and mark the subsystem as stopped."""
        self.last_error = message
        self.is_running = False
        print(f"Error in audio capture: {message}")

    def _find_loopback_mic(self):
        """Locate the loopback microphone matching the default speaker, if any."""
        default_speaker = sc.default_speaker()
        mics = sc.all_microphones(include_loopback=True)

        # Find the loopback microphone for the default speaker
        for mic in mics:
            if mic.isloopback and default_speaker.id in mic.id:
                return default_speaker, mic

        # Fallback to the default loopback if specific one not found
        for mic in mics:
            if mic.isloopback and default_speaker.name in mic.name:
                return default_speaker, mic

        # Absolute fallback: just get the first loopback
        for mic in mics:
            if mic.isloopback:
                return default_speaker, mic

        return default_speaker, None

    def _capture_loop(self):
        try:
            default_speaker, loopback_mic = self._find_loopback_mic()

            print(f"Using default speaker: {default_speaker.name}")

            if loopback_mic is None:
                self._fail("No loopback-capable microphone found. System audio "
                           "capture requires a loopback/monitor device.")
                return

            print(f"Selected loopback mic: {loopback_mic.name}")

            with loopback_mic.recorder(samplerate=self.sample_rate) as mic:
                while self.is_running:
                    # Record a chunk of audio
                    data = mic.record(numframes=self.buffer_frames)

                    # Convert to mono by averaging channels. Backends may hand
                    # back an already-flat array for a single-channel device.
                    mono_data = data.mean(axis=1) if data.ndim > 1 else data

                    # Put in deque, automatically discards oldest if full (maxlen=1)
                    self.audio_queue.append(mono_data)
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")

    def get_latest_data(self):
        """Retrieve the most recent audio chunk, if available."""
        try:
            return self.audio_queue.pop()
        except IndexError:
            return None
