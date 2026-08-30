"""
Local audio file loading and playback.

`AudioCaptureSubsystem` visualizes whatever the system is playing. This module
covers the other direction: the user picks a file, we decode it, push it to the
default speaker ourselves, and hand the very samples we wrote to the sound card
to the visualizer. Visualizing the decoded samples rather than re-capturing the
speaker output keeps the picture in step with the audio and works even on
machines with no loopback device.

Decoding goes through libsndfile (the `soundfile` package), which covers wav,
flac, mp3, ogg, aiff and friends. Formats it does not know (m4a, aac, wma, ...)
are handed to ffmpeg when it is installed on the machine.
"""

import os
import shutil
import subprocess
import threading
from collections import deque

import numpy as np

# Extensions offered in the file dialog. Decoding is not actually driven by the
# extension -- libsndfile is tried first for any file and ffmpeg picks up what
# it rejects -- so this list only has to be a good default for the picker.
SUPPORTED_EXTENSIONS = (
    ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac",
    ".aiff", ".aif", ".aifc", ".wma", ".alac", ".ape", ".mp4", ".mka",
    ".webm", ".au", ".caf", ".w64", ".wv",
)

# Frames handed to the speaker (and to the visualizer) at a time. At 44.1 kHz
# this is ~46 ms, short enough for pause/seek to feel immediate and to keep the
# published chunk close to what is actually audible.
DEFAULT_CHUNK_FRAMES = 2048

# How long stop()/unload() waits for the playback thread to unwind before
# giving up on it, mirroring AudioCaptureSubsystem.JOIN_TIMEOUT_SECONDS.
JOIN_TIMEOUT_SECONDS = 2.0

# How long the playback thread sleeps between checks while paused. Also the
# rate at which it publishes silence so the visualizer settles instead of
# freezing on the last frame before the pause.
PAUSED_POLL_SECONDS = 0.05

# Speakers are commonly stereo; more channels than this get mixed down rather
# than risking a device that cannot open the file's channel count.
MAX_OUTPUT_CHANNELS = 2

# Upper bound on an ffmpeg decode, so a hung or malicious file cannot wedge the
# UI thread that is waiting for the load to finish.
FFMPEG_TIMEOUT_SECONDS = 300

# Sample rate requested from ffmpeg when ffprobe cannot tell us the file's own.
FFMPEG_FALLBACK_RATE = 44100


class AudioFileError(Exception):
    """Raised when a file cannot be read or decoded."""


def file_dialog_filter():
    """Qt file-dialog filter string covering the formats we try to open."""
    patterns = " ".join("*" + ext for ext in SUPPORTED_EXTENSIONS)
    return f"Audio files ({patterns});;All files (*)"


def load_audio_file(path):
    """
    Decode `path` into (samples, sample_rate).

    `samples` is a float32 array of shape (frames, channels); `sample_rate` is
    the file's own rate -- nothing is resampled, so playback is bit-accurate
    and the caller retunes its analysis to the returned rate instead.

    Raises AudioFileError with a message fit for display if the file cannot be
    read by libsndfile or by ffmpeg.
    """
    if not os.path.isfile(path):
        raise AudioFileError(f"{os.path.basename(path)}: file not found.")

    reasons = []

    try:
        import soundfile
    except Exception as exc:  # pragma: no cover - soundfile is a hard dependency
        reasons.append(f"soundfile unavailable ({exc})")
    else:
        try:
            samples, sample_rate = soundfile.read(path, dtype="float32",
                                                  always_2d=True)
            return _validate(samples, int(sample_rate), path)
        except Exception as exc:
            reasons.append(f"libsndfile: {exc}")

    try:
        samples, sample_rate = _decode_with_ffmpeg(path)
    except AudioFileError as exc:
        reasons.append(str(exc))
    else:
        return _validate(samples, sample_rate, path)

    raise AudioFileError(
        f"Could not open {os.path.basename(path)} — " + "; ".join(reasons))


def _validate(samples, sample_rate, path):
    """Reject decodes that would leave the player with nothing to play."""
    name = os.path.basename(path)
    if sample_rate <= 0:
        raise AudioFileError(f"{name}: reported an invalid sample rate.")
    if samples.size == 0 or samples.shape[0] == 0:
        raise AudioFileError(f"{name}: contains no audio.")
    # NaN/Inf in a decoded stream would reach both the sound card and the FFT.
    if not np.all(np.isfinite(samples)):
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(samples, dtype=np.float32), sample_rate


def _ffprobe_sample_rate(path):
    """The file's sample rate according to ffprobe, or None if unavailable."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        rate = int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None
    return rate if rate > 0 else None


def _decode_with_ffmpeg(path):
    """Decode `path` to float32 stereo PCM using the system ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioFileError(
            "ffmpeg is not installed (needed for this format)")

    sample_rate = _ffprobe_sample_rate(path) or FFMPEG_FALLBACK_RATE
    try:
        result = subprocess.run(
            [ffmpeg, "-v", "error", "-i", path,
             "-f", "f32le", "-acodec", "pcm_f32le",
             "-ac", str(MAX_OUTPUT_CHANNELS), "-ar", str(sample_rate), "-"],
            capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        raise AudioFileError("ffmpeg: timed out decoding the file")
    except OSError as exc:
        raise AudioFileError(f"ffmpeg: {exc}")

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioFileError(f"ffmpeg: {detail[-1] if detail else 'decode failed'}")

    raw = np.frombuffer(result.stdout, dtype="<f4")
    # A stream cut short mid-frame would break the reshape below.
    usable = (raw.size // MAX_OUTPUT_CHANNELS) * MAX_OUTPUT_CHANNELS
    if usable == 0:
        raise AudioFileError("ffmpeg: produced no audio")
    return raw[:usable].reshape(-1, MAX_OUTPUT_CHANNELS), sample_rate


def _default_speaker():
    """The system's default output device.

    Imported lazily: `soundcard` binds to the platform audio stack at import
    time, and this module must stay importable (and testable) without one.
    """
    import soundcard as sc
    return sc.default_speaker()


class AudioFilePlayer:
    """
    Plays a decoded audio file and publishes what it plays for visualization.

    The consumer side matches AudioCaptureSubsystem -- `get_latest_data()`,
    `is_active()`, `last_error` -- so the application can swap between a live
    capture and a file without special-casing the render loop.

    Playback runs on a background thread; `play`, `pause`, `seek` and `unload`
    are safe to call from the UI thread and take effect within one chunk.
    """

    def __init__(self, chunk_frames=DEFAULT_CHUNK_FRAMES, speaker_factory=None):
        self.chunk_frames = chunk_frames
        self.audio_queue = deque(maxlen=1)
        self.last_error = None
        self.path = None
        self.sample_rate = 0

        self._speaker_factory = speaker_factory or _default_speaker
        self._playable = None   # (frames, channels) float32, as sent to the device
        self._mono = None       # mono mix of the same frames, for the FFT
        self._silence = np.zeros(chunk_frames)

        # `_frame` is the next frame to play. Only the playback thread advances
        # it; seek() rewrites it, and the thread notices via the value check in
        # _advance() so a seek is never overwritten by the chunk in flight.
        self._frame = 0
        self._lock = threading.Lock()

        self._resume = threading.Event()
        self._is_playing = False
        self._running = False
        self._thread = None

    # ── Loading ──

    def load(self, path):
        """
        Decode `path` and begin playing it.

        Raises AudioFileError if the file cannot be decoded, leaving any
        previously loaded file unloaded.
        """
        samples, sample_rate = load_audio_file(path)

        self.unload()

        self._playable = _mix_for_output(samples)
        self._mono = self._playable.mean(axis=1)
        self.sample_rate = sample_rate
        self.path = path
        self._frame = 0
        self.last_error = None
        self.audio_queue.clear()

        self._running = True
        self._is_playing = True
        self._resume.set()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def unload(self):
        """Stop playback and forget the current file."""
        self._running = False
        self._is_playing = False
        # Wake a paused thread so it can observe _running and exit.
        self._resume.set()
        if self._thread is not None:
            self._thread.join(timeout=JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                print("Warning: audio playback thread did not stop within "
                      f"{JOIN_TIMEOUT_SECONDS}s; abandoning it.")
            self._thread = None
        self._playable = None
        self._mono = None
        self.path = None
        self.sample_rate = 0
        self._frame = 0
        self.audio_queue.clear()

    # ── Transport ──

    @property
    def has_file(self):
        return self._playable is not None

    @property
    def is_playing(self):
        return self._is_playing

    @property
    def duration(self):
        """Length of the loaded file in seconds (0.0 when nothing is loaded)."""
        if not self.has_file or self.sample_rate <= 0:
            return 0.0
        return len(self._playable) / self.sample_rate

    @property
    def position(self):
        """Playback position in seconds."""
        if not self.has_file or self.sample_rate <= 0:
            return 0.0
        return self._frame / self.sample_rate

    def play(self):
        """Resume playback, restarting from the top if the file has ended."""
        if not self.has_file or not self._running:
            return
        with self._lock:
            if self._frame >= len(self._playable):
                self._frame = 0
        self._is_playing = True
        self._resume.set()

    def pause(self):
        if not self.has_file:
            return
        self._is_playing = False
        self._resume.clear()

    def toggle(self):
        """Flip between playing and paused; returns the new playing state."""
        if self._is_playing:
            self.pause()
        else:
            self.play()
        return self._is_playing

    def seek(self, seconds):
        """Jump to `seconds` into the file, clamped to its bounds."""
        if not self.has_file or self.sample_rate <= 0:
            return
        frame = int(max(0.0, seconds) * self.sample_rate)
        with self._lock:
            self._frame = min(frame, len(self._playable))

    # ── Consumer interface (mirrors AudioCaptureSubsystem) ──

    def is_active(self):
        """True while a file is loaded and playback has not failed."""
        return self._running and self.last_error is None

    def get_latest_data(self):
        """Retrieve the most recent audio chunk, if available."""
        try:
            return self.audio_queue.pop()
        except IndexError:
            return None

    # ── Playback thread ──

    def _fail(self, message):
        self.last_error = message
        self._running = False
        self._is_playing = False
        print(f"Error in audio playback: {message}")

    def _advance(self, start, end):
        """
        Commit progress from `start` to `end`, unless a seek moved us.

        Returns True if the file has now been played to its end.
        """
        with self._lock:
            if self._frame == start:
                self._frame = end
            return self._frame >= len(self._playable)

    def _playback_loop(self):
        try:
            speaker = self._speaker_factory()
            if speaker is None:
                self._fail("No audio output device is available.")
                return

            channels = self._playable.shape[1]
            with speaker.player(samplerate=self.sample_rate,
                                channels=channels) as player:
                while self._running:
                    if not self._is_playing:
                        # Publish silence so the visualizer decays to rest
                        # instead of holding the frame the pause landed on.
                        self.audio_queue.append(self._silence)
                        self._resume.wait(PAUSED_POLL_SECONDS)
                        continue

                    with self._lock:
                        start = self._frame
                        end = min(start + self.chunk_frames, len(self._playable))
                    if end <= start:
                        self.pause()
                        continue

                    player.play(self._playable[start:end])
                    # Published after the write so the picture tracks what the
                    # device has actually taken, not what we queued ahead.
                    self.audio_queue.append(self._mono[start:end])

                    if self._advance(start, end):
                        self.pause()
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")


def _mix_for_output(samples):
    """
    Shape decoded samples into something the default speaker will accept.

    Mono is widened to stereo and anything above stereo is mixed down, so we
    never ask a device to open a channel count it does not have.
    """
    frames, channels = samples.shape
    if channels == 1:
        return np.repeat(samples, MAX_OUTPUT_CHANNELS, axis=1)
    if channels <= MAX_OUTPUT_CHANNELS:
        return samples
    # Fold the extra channels into both outputs rather than dropping them,
    # which would silence anything mixed only into the surround channels.
    mono = samples.mean(axis=1, keepdims=True)
    return np.repeat(mono.astype(np.float32), MAX_OUTPUT_CHANNELS, axis=1)
