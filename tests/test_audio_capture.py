"""
Unit tests for the AudioCaptureSubsystem.

`soundcard` is stubbed out: it binds to a live audio stack at import time, which
CI machines do not have, and these tests are about the subsystem's own device
selection, error handling and shutdown behaviour.
"""
import importlib
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

# audio_capture imports soundcard at module scope. Importing the real library
# binds to the system audio stack (libpulse), which fails on a headless CI box,
# so install a stub when it cannot be loaded.
if "soundcard" not in sys.modules:
    try:
        importlib.import_module("soundcard")
    except Exception:
        sys.modules["soundcard"] = types.ModuleType("soundcard")

import audio_capture
from audio_capture import AudioCaptureSubsystem


class FakeSpeaker:
    def __init__(self, name="Speakers", id_="spk-1"):
        self.name = name
        self.id = id_


class FakeRecorder:
    """Context manager standing in for soundcard's recorder."""

    def __init__(self, mic):
        self.mic = mic

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def record(self, numframes):
        if self.mic.block_event is not None:
            # Simulate a device that never returns another buffer.
            self.mic.block_event.wait()
        if self.mic.error is not None:
            raise self.mic.error
        self.mic.record_count += 1
        return self.mic.frames


class FakeMic:
    def __init__(self, name, id_, isloopback=True, frames=None,
                 error=None, block_event=None):
        self.name = name
        self.id = id_
        self.isloopback = isloopback
        self.frames = frames if frames is not None else np.zeros((1024, 2))
        self.error = error
        self.block_event = block_event
        self.record_count = 0

    def recorder(self, samplerate):
        return FakeRecorder(self)


def fake_soundcard(speaker, mics):
    module = mock.Mock()
    module.default_speaker.return_value = speaker
    module.all_microphones.return_value = mics
    return module


class CaptureTestCase(unittest.TestCase):
    def run_capture(self, speaker, mics, until, timeout=2.0):
        """Start a capture against fake devices and wait for `until()`."""
        capture = AudioCaptureSubsystem(sample_rate=44100, buffer_frames=1024)
        with mock.patch.object(audio_capture, "sc", fake_soundcard(speaker, mics)):
            capture.start()
            self.addCleanup(capture.stop)
            deadline = time.time() + timeout
            while time.time() < deadline and not until(capture):
                time.sleep(0.005)
        return capture


class TestDeviceSelection(CaptureTestCase):
    """The loopback fallback chain must not dereference a missing device."""

    def test_no_loopback_device_reports_error_instead_of_crashing(self):
        """
        Regression: the None check printed 'None' and then called
        loopback_mic.recorder() anyway, so the AttributeError was swallowed by
        the blanket except and the UI showed a permanently frozen visualizer.
        """
        speaker = FakeSpeaker()
        mics = [FakeMic("Built-in Mic", "mic-1", isloopback=False)]

        capture = self.run_capture(speaker, mics,
                                   until=lambda c: c.last_error is not None)

        self.assertIsNotNone(capture.last_error)
        self.assertIn("loopback", capture.last_error.lower())
        self.assertNotIn("NoneType", capture.last_error)
        self.assertFalse(capture.is_running)
        self.assertFalse(capture.is_active())
        self.assertIsNone(capture.get_latest_data())

    def test_prefers_loopback_matching_the_default_speaker_id(self):
        speaker = FakeSpeaker(name="Speakers", id_="spk-1")
        wanted = FakeMic("Other name", "loopback-spk-1")
        mics = [FakeMic("Speakers (loopback)", "other-id"), wanted]

        capture = self.run_capture(speaker, mics,
                                   until=lambda c: wanted.record_count > 0)
        self.assertGreater(wanted.record_count, 0)
        self.assertIsNone(capture.last_error)

    def test_falls_back_to_name_match_then_any_loopback(self):
        speaker = FakeSpeaker(name="Speakers", id_="spk-1")
        by_name = FakeMic("Speakers (loopback)", "unrelated-id")
        capture = self.run_capture(speaker, [by_name],
                                   until=lambda c: by_name.record_count > 0)
        self.assertGreater(by_name.record_count, 0)

        any_loopback = FakeMic("Monitor of Something", "unrelated-id")
        capture = self.run_capture(speaker, [any_loopback],
                                   until=lambda c: any_loopback.record_count > 0)
        self.assertGreater(any_loopback.record_count, 0)
        self.assertIsNone(capture.last_error)


class TestCaptureLoop(CaptureTestCase):
    def test_stereo_is_averaged_to_mono(self):
        frames = np.tile(np.array([[0.0, 1.0]]), (1024, 1))
        mic = FakeMic("loopback", "spk-1", frames=frames)

        # Note: get_latest_data() pops, so poll the queue length instead.
        capture = self.run_capture(FakeSpeaker(), [mic],
                                   until=lambda c: len(c.audio_queue) > 0)
        data = capture.get_latest_data()
        self.assertIsNotNone(data)
        self.assertEqual(data.shape, (1024,))
        np.testing.assert_allclose(data, 0.5)

    def test_single_channel_input_is_not_reduced(self):
        """
        A 1-D buffer must pass through rather than raising AxisError inside
        data.mean(axis=1) and killing the capture thread.
        """
        frames = np.full(1024, 0.25)
        mic = FakeMic("loopback", "spk-1", frames=frames)

        capture = self.run_capture(FakeSpeaker(), [mic],
                                   until=lambda c: len(c.audio_queue) > 0)
        self.assertIsNone(capture.last_error)
        np.testing.assert_allclose(capture.audio_queue[0], 0.25)

    def test_device_failure_is_recorded_and_stops_the_subsystem(self):
        """
        Regression: a failing capture left is_running True, so a dead thread was
        indistinguishable from a working one and nothing could detect it.
        """
        mic = FakeMic("loopback", "spk-1", error=RuntimeError("device removed"))

        capture = self.run_capture(FakeSpeaker(), [mic],
                                   until=lambda c: c.last_error is not None)

        self.assertIn("device removed", capture.last_error)
        self.assertIn("RuntimeError", capture.last_error)
        self.assertFalse(capture.is_running)
        self.assertFalse(capture.is_active())

    def test_healthy_capture_reports_active(self):
        mic = FakeMic("loopback", "spk-1")
        capture = self.run_capture(FakeSpeaker(), [mic],
                                   until=lambda c: mic.record_count > 0)
        self.assertTrue(capture.is_active())
        self.assertIsNone(capture.last_error)


class TestShutdown(unittest.TestCase):
    def test_stop_does_not_hang_on_a_wedged_device(self):
        """
        Regression: stop() joined with no timeout while the thread was blocked
        inside record(), so closing the window deadlocked the process.
        """
        blocked = threading.Event()
        self.addCleanup(blocked.set)
        mic = FakeMic("loopback", "spk-1", block_event=blocked)

        capture = AudioCaptureSubsystem(sample_rate=44100, buffer_frames=1024)
        with mock.patch.object(audio_capture, "sc",
                               fake_soundcard(FakeSpeaker(), [mic])), \
             mock.patch.object(audio_capture, "JOIN_TIMEOUT_SECONDS", 0.2):
            capture.start()
            # Let the thread reach the blocking record() call.
            time.sleep(0.1)

            started = time.time()
            capture.stop()
            elapsed = time.time() - started

        self.assertLess(elapsed, 1.5, "stop() blocked on the wedged thread")
        self.assertFalse(capture.is_running)

    def test_stop_joins_a_healthy_thread(self):
        mic = FakeMic("loopback", "spk-1")
        capture = AudioCaptureSubsystem(sample_rate=44100, buffer_frames=1024)
        with mock.patch.object(audio_capture, "sc",
                               fake_soundcard(FakeSpeaker(), [mic])):
            capture.start()
            deadline = time.time() + 2.0
            while time.time() < deadline and mic.record_count == 0:
                time.sleep(0.005)
            capture.stop()

        self.assertFalse(capture.is_running)
        self.assertIsNone(capture.capture_thread)

    def test_stop_without_start_is_a_no_op(self):
        AudioCaptureSubsystem().stop()


if __name__ == '__main__':
    unittest.main()
