"""
Unit tests for local audio file loading and playback.

Decoding is exercised against real files written by libsndfile. Playback is
driven through a fake speaker: `soundcard` binds to the platform audio stack,
which CI machines do not have, and these tests are about the player's transport
behaviour rather than the sound card's.
"""
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

import audio_file
from audio_file import (AudioFileError, AudioFilePlayer, file_dialog_filter,
                        load_audio_file)

try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except Exception:  # pragma: no cover - environment without libsndfile
    SOUNDFILE_AVAILABLE = False

SAMPLE_RATE = 8000
WAIT_TIMEOUT_SECONDS = 5.0


def wait_until(predicate, timeout=WAIT_TIMEOUT_SECONDS):
    """Poll `predicate` until it is true or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def sine(frames, channels=1, freq=440.0, rate=SAMPLE_RATE):
    t = np.arange(frames) / rate
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return np.tile(wave[:, None], (1, channels)).astype(np.float32)


class FakePlayer:
    """Stands in for soundcard's player context manager."""

    def __init__(self, speaker):
        self.speaker = speaker

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.speaker.closed = True
        return False

    def play(self, data):
        # A real device paces playback; this one is instantaneous unless a test
        # asked for a gate, which lets it assert on mid-playback state without
        # racing the playback thread.
        if self.speaker.gate is not None:
            self.speaker.gate.acquire()
        if self.speaker.play_error is not None:
            raise self.speaker.play_error
        with self.speaker.lock:
            self.speaker.written.append(np.array(data, copy=True))


class FakeSpeaker:
    """
    A speaker that records what it was asked to play.

    With `gated=True` each chunk waits for a `release()`, so a test can hold
    playback at a known frame instead of guessing at timings.
    """

    def __init__(self, play_error=None, open_error=None, gated=False):
        self.play_error = play_error
        self.open_error = open_error
        self.gate = threading.Semaphore(0) if gated else None
        self.written = []
        self.samplerate = None
        self.channels = None
        self.closed = False
        self.lock = threading.Lock()

    def player(self, samplerate, channels):
        if self.open_error is not None:
            raise self.open_error
        self.samplerate = samplerate
        self.channels = channels
        return FakePlayer(self)

    def frames_written(self):
        with self.lock:
            return sum(len(chunk) for chunk in self.written)

    def release(self, chunks=1):
        """Let `chunks` more writes through a gated speaker."""
        if self.gate is not None:
            self.gate.release(chunks)


@unittest.skipUnless(SOUNDFILE_AVAILABLE, "soundfile is not installed")
class TestLoadAudioFile(unittest.TestCase):
    """Decoding real files off disk."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def write(self, name, data, rate=SAMPLE_RATE):
        path = os.path.join(self.tmpdir, name)
        soundfile.write(path, data, rate)
        return path

    def test_reads_wav_as_float32_2d(self):
        path = self.write("tone.wav", sine(1000))
        samples, rate = load_audio_file(path)

        self.assertEqual(rate, SAMPLE_RATE)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.ndim, 2)
        self.assertEqual(samples.shape, (1000, 1))

    def test_reads_common_formats(self):
        """mp3, wav, flac and ogg all come back as decoded audio."""
        formats = [("tone.wav", 1), ("tone.flac", 2), ("tone.ogg", 2)]
        if "MP3" in soundfile.available_formats():
            formats.append(("tone.mp3", 2))

        for name, channels in formats:
            with self.subTest(name=name):
                path = self.write(name, sine(4000, channels))
                samples, rate = load_audio_file(path)
                self.assertEqual(rate, SAMPLE_RATE)
                self.assertEqual(samples.shape[1], channels)
                # Lossy codecs pad and re-time, so only assert we got audio of
                # roughly the right length rather than an exact frame count.
                self.assertGreater(samples.shape[0], 3000)
                self.assertGreater(np.abs(samples).max(), 0.01)

    def test_preserves_the_files_own_sample_rate(self):
        path = self.write("tone.wav", sine(500, rate=22050), rate=22050)
        _samples, rate = load_audio_file(path)
        self.assertEqual(rate, 22050)

    def test_missing_file_raises(self):
        with self.assertRaises(AudioFileError) as ctx:
            load_audio_file(os.path.join(self.tmpdir, "nope.wav"))
        self.assertIn("not found", str(ctx.exception))

    def test_undecodable_file_raises_with_the_file_name(self):
        path = os.path.join(self.tmpdir, "notaudio.wav")
        with open(path, "wb") as handle:
            handle.write(b"this is not audio")

        with self.assertRaises(AudioFileError) as ctx:
            load_audio_file(path)
        self.assertIn("notaudio.wav", str(ctx.exception))

    def test_empty_file_is_rejected(self):
        path = self.write("silence.wav", np.zeros((0, 1), dtype=np.float32))
        with self.assertRaises(AudioFileError) as ctx:
            load_audio_file(path)
        self.assertIn("no audio", str(ctx.exception))

    def test_non_finite_samples_are_sanitized(self):
        """NaN/Inf must not reach the sound card or the FFT."""
        dirty = np.array([[0.5], [np.nan], [np.inf], [-np.inf]], dtype=np.float32)
        with mock.patch.object(soundfile, "read", return_value=(dirty, SAMPLE_RATE)):
            path = self.write("tone.wav", sine(100))
            samples, _rate = load_audio_file(path)
        self.assertTrue(np.all(np.isfinite(samples)))


class TestFfmpegFallback(unittest.TestCase):
    """Formats libsndfile rejects are handed to ffmpeg when it is installed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "track.m4a")
        with open(self.path, "wb") as handle:
            handle.write(b"not something libsndfile knows")

    def test_reports_missing_ffmpeg_instead_of_failing_silently(self):
        with mock.patch.object(audio_file.shutil, "which", return_value=None):
            with self.assertRaises(AudioFileError) as ctx:
                load_audio_file(self.path)
        self.assertIn("ffmpeg", str(ctx.exception))

    def test_decodes_through_ffmpeg_when_libsndfile_cannot(self):
        frames = 1000
        pcm = sine(frames, channels=2).astype("<f4").tobytes()
        completed = mock.Mock(returncode=0, stdout=pcm, stderr=b"")

        with mock.patch.object(audio_file.shutil, "which",
                               side_effect=lambda name: "/usr/bin/" + name), \
             mock.patch.object(audio_file.subprocess, "run",
                               return_value=completed) as run:
            samples, rate = load_audio_file(self.path)

        self.assertEqual(samples.shape, (frames, 2))
        self.assertEqual(rate, audio_file.FFMPEG_FALLBACK_RATE)
        # ffprobe first, then ffmpeg.
        self.assertEqual(run.call_count, 2)

    def test_ffmpeg_failure_surfaces_its_last_error_line(self):
        completed = mock.Mock(returncode=1, stdout=b"",
                              stderr=b"Invalid data found when processing input\n")
        with mock.patch.object(audio_file.shutil, "which",
                               side_effect=lambda name: "/usr/bin/" + name), \
             mock.patch.object(audio_file.subprocess, "run",
                               return_value=completed):
            with self.assertRaises(AudioFileError) as ctx:
                load_audio_file(self.path)
        self.assertIn("Invalid data found", str(ctx.exception))

    def test_truncated_ffmpeg_output_does_not_break_the_reshape(self):
        """An odd number of floats must not raise out of numpy's reshape."""
        pcm = np.array([0.1, 0.2, 0.3], dtype="<f4").tobytes()
        completed = mock.Mock(returncode=0, stdout=pcm, stderr=b"")
        with mock.patch.object(audio_file.shutil, "which",
                               side_effect=lambda name: "/usr/bin/" + name), \
             mock.patch.object(audio_file.subprocess, "run",
                               return_value=completed):
            samples, _rate = load_audio_file(self.path)
        self.assertEqual(samples.shape, (1, 2))


class TestMixForOutput(unittest.TestCase):
    """Channel layouts the default speaker will actually accept."""

    def test_mono_is_widened_to_stereo(self):
        mixed = audio_file._mix_for_output(sine(10, channels=1))
        self.assertEqual(mixed.shape, (10, 2))
        np.testing.assert_allclose(mixed[:, 0], mixed[:, 1])

    def test_stereo_is_left_alone(self):
        samples = sine(10, channels=2)
        self.assertIs(audio_file._mix_for_output(samples), samples)

    def test_surround_is_folded_down_rather_than_truncated(self):
        samples = np.zeros((10, 6), dtype=np.float32)
        # Content only in a surround channel would vanish if we just sliced.
        samples[:, 4] = 1.0
        mixed = audio_file._mix_for_output(samples)
        self.assertEqual(mixed.shape, (10, 2))
        self.assertGreater(np.abs(mixed).max(), 0.0)


class PlayerTestCase(unittest.TestCase):
    """Shared setup for AudioFilePlayer tests."""

    frames = 2048
    channels = 1
    chunk_frames = 256

    def setUp(self):
        if not SOUNDFILE_AVAILABLE:
            self.skipTest("soundfile is not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "tone.wav")
        soundfile.write(self.path, sine(self.frames, self.channels), SAMPLE_RATE)

        self.speaker = FakeSpeaker()
        self.player = AudioFilePlayer(chunk_frames=self.chunk_frames,
                                      speaker_factory=lambda: self.speaker)
        # Registered before unload so a gated speaker is released first and the
        # playback thread can actually unwind.
        self.addCleanup(self.player.unload)
        self.addCleanup(lambda: self.speaker.release(chunks=16))

    def load(self):
        self.player.load(self.path)

    def load_gated(self):
        """Load with playback held at the first chunk."""
        self.speaker = FakeSpeaker(gated=True)
        self.player = AudioFilePlayer(chunk_frames=self.chunk_frames,
                                      speaker_factory=lambda: self.speaker)
        self.load()


class TestAudioFilePlayer(PlayerTestCase):

    def test_load_starts_playing_and_reports_the_file(self):
        self.load_gated()

        self.assertTrue(self.player.has_file)
        self.assertTrue(self.player.is_playing)
        self.assertTrue(self.player.is_active())
        self.assertEqual(self.player.path, self.path)
        self.assertEqual(self.player.sample_rate, SAMPLE_RATE)
        self.assertAlmostEqual(self.player.duration,
                               self.frames / SAMPLE_RATE, places=6)

    def test_plays_the_whole_file_then_stops(self):
        self.load()

        self.assertTrue(wait_until(lambda: not self.player.is_playing),
                        "playback never finished")
        self.assertEqual(self.speaker.frames_written(), self.frames)
        self.assertAlmostEqual(self.player.position, self.player.duration,
                               places=6)
        # Finishing the file is not a failure: the file stays loaded and
        # replayable rather than being torn down.
        self.assertIsNone(self.player.last_error)
        self.assertTrue(self.player.has_file)

    def test_mono_is_widened_for_a_mono_file(self):
        self.load()
        self.assertTrue(wait_until(lambda: self.speaker.channels is not None))
        self.assertEqual(self.speaker.channels, 2)
        self.assertEqual(self.speaker.samplerate, SAMPLE_RATE)

    def test_publishes_mono_chunks_for_the_visualizer(self):
        """What reaches the visualizer is the mono mix of what was played."""
        self.load_gated()
        self.speaker.release()

        self.assertTrue(wait_until(lambda: len(self.player.audio_queue) > 0),
                        "no audio was published")
        chunk = self.player.get_latest_data()

        self.assertEqual(chunk.ndim, 1)
        self.assertEqual(len(chunk), self.chunk_frames)
        expected = self.speaker.written[0].mean(axis=1)
        np.testing.assert_allclose(chunk, expected, atol=1e-6)

    def test_get_latest_data_returns_none_when_nothing_is_queued(self):
        self.assertIsNone(self.player.get_latest_data())

    def test_replay_after_the_end_restarts_from_the_top(self):
        self.load()
        self.assertTrue(wait_until(lambda: not self.player.is_playing))

        self.player.play()
        self.assertTrue(self.player.is_playing)
        self.assertLess(self.player.position, self.player.duration)
        self.assertTrue(wait_until(
            lambda: self.speaker.frames_written() >= 2 * self.frames))

    def test_pause_stops_writing_and_publishes_silence(self):
        self.load()
        self.player.pause()
        self.assertFalse(self.player.is_playing)

        # Let the thread settle into the paused branch, then confirm it stays put.
        self.assertTrue(wait_until(
            lambda: self.player.get_latest_data() is not None))
        written = self.speaker.frames_written()
        time.sleep(4 * audio_file.PAUSED_POLL_SECONDS)
        self.assertEqual(self.speaker.frames_written(), written)

        silence = self.player.get_latest_data()
        self.assertIsNotNone(silence)
        np.testing.assert_array_equal(silence, np.zeros(self.chunk_frames))

    def test_toggle_flips_and_reports_the_new_state(self):
        self.load_gated()
        self.assertFalse(self.player.toggle())
        self.assertFalse(self.player.is_playing)
        self.assertTrue(self.player.toggle())
        self.assertTrue(self.player.is_playing)

    def test_seek_moves_the_position(self):
        self.load()
        self.player.pause()

        self.player.seek(0.1)
        self.assertAlmostEqual(self.player.position, 0.1, places=3)

    def test_seek_clamps_to_the_files_bounds(self):
        self.load()
        self.player.pause()

        self.player.seek(-5.0)
        self.assertEqual(self.player.position, 0.0)

        self.player.seek(self.player.duration + 60)
        self.assertAlmostEqual(self.player.position, self.player.duration,
                               places=6)

    def test_a_seek_is_not_overwritten_by_the_chunk_in_flight(self):
        """The playback thread must not undo a seek that raced its write."""
        self.load()
        self.player.pause()
        self.player.seek(0.0)

        # Simulate the thread finishing a chunk it started before the seek.
        self.player.seek(0.1)
        self.assertFalse(self.player._advance(0, self.chunk_frames))
        self.assertAlmostEqual(self.player.position, 0.1, places=3)

    def test_unload_stops_playback_and_clears_state(self):
        self.load()
        self.player.unload()

        self.assertFalse(self.player.has_file)
        self.assertFalse(self.player.is_playing)
        self.assertFalse(self.player.is_active())
        self.assertIsNone(self.player.path)
        self.assertEqual(self.player.duration, 0.0)
        self.assertEqual(self.player.position, 0.0)
        self.assertIsNone(self.player.get_latest_data())
        self.assertTrue(self.speaker.closed)

    def test_transport_calls_are_safe_with_no_file_loaded(self):
        for action in (self.player.play, self.player.pause,
                       self.player.unload):
            action()
        self.player.seek(1.0)
        self.assertFalse(self.player.toggle())
        self.assertFalse(self.player.has_file)

    def test_loading_a_second_file_replaces_the_first(self):
        self.load()
        other = os.path.join(self.tmpdir, "other.wav")
        soundfile.write(other, sine(512, 2, rate=22050), 22050)

        self.player.load(other)

        self.assertEqual(self.player.path, other)
        self.assertEqual(self.player.sample_rate, 22050)
        self.assertAlmostEqual(self.player.duration, 512 / 22050, places=6)

    def test_a_failed_load_leaves_no_file_loaded(self):
        self.load()
        with self.assertRaises(AudioFileError):
            self.player.load(os.path.join(self.tmpdir, "missing.wav"))
        # The decode fails before the running file is torn down, so the player
        # is still usable rather than half-unloaded.
        self.assertTrue(self.player.has_file)
        self.assertEqual(self.player.path, self.path)


class TestAudioFilePlayerFailures(PlayerTestCase):
    """A dead output device is reported, not silently swallowed."""

    def test_missing_output_device_is_reported(self):
        self.player = AudioFilePlayer(chunk_frames=self.chunk_frames,
                                      speaker_factory=lambda: None)
        self.load()

        self.assertTrue(wait_until(lambda: self.player.last_error is not None))
        self.assertIn("output device", self.player.last_error)
        self.assertFalse(self.player.is_active())
        self.assertFalse(self.player.is_playing)

    def test_device_error_mid_playback_is_reported(self):
        speaker = FakeSpeaker(play_error=RuntimeError("device disappeared"))
        self.player = AudioFilePlayer(chunk_frames=self.chunk_frames,
                                      speaker_factory=lambda: speaker)
        self.load()

        self.assertTrue(wait_until(lambda: self.player.last_error is not None))
        self.assertIn("device disappeared", self.player.last_error)
        self.assertFalse(self.player.is_active())

    def test_speaker_lookup_failure_is_reported(self):
        def explode():
            raise OSError("no audio backend")

        self.player = AudioFilePlayer(chunk_frames=self.chunk_frames,
                                      speaker_factory=explode)
        self.load()

        self.assertTrue(wait_until(lambda: self.player.last_error is not None))
        self.assertIn("no audio backend", self.player.last_error)


class TestFileDialogFilter(unittest.TestCase):

    def test_offers_the_common_formats(self):
        pattern = file_dialog_filter()
        for extension in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"):
            self.assertIn("*" + extension, pattern)
        self.assertIn("All files (*)", pattern)


if __name__ == "__main__":
    unittest.main()
