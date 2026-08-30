"""
End-to-end tests for AudioVisualizerApp's source handling.

A real file is decoded by the real AudioFilePlayer and pushed through the real
AudioProcessor and MainWindow; only the two things that need hardware are
faked -- the system capture and the speaker the player writes to.
"""
import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# main.py imports audio_capture, which imports soundcard at module scope and so
# binds to the platform audio stack. Stub it when that is not available, as
# tests/test_audio_capture.py does.
if "soundcard" not in sys.modules:
    try:
        importlib.import_module("soundcard")
    except Exception:
        sys.modules["soundcard"] = types.ModuleType("soundcard")

try:
    from PyQt6.QtWidgets import QApplication
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without PyQt6
    PYQT_AVAILABLE = False

try:
    import soundfile
    SOUNDFILE_AVAILABLE = True
except Exception:  # pragma: no cover - environment without libsndfile
    SOUNDFILE_AVAILABLE = False

if PYQT_AVAILABLE:
    import main
    from audio_file import AudioFilePlayer

from tests.test_audio_file import FakeSpeaker, wait_until

_app = None

FILE_RATE = 22050
FILE_FRAMES = 8192


def setUpModule():
    global _app
    if PYQT_AVAILABLE:
        _app = QApplication.instance() or QApplication([])


class FakeCapture:
    """Records start/stop without touching a device."""

    def __init__(self, *_args, **_kwargs):
        self.last_error = None
        self.started = 0
        self.stopped = 0
        self.chunks = []

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def get_latest_data(self):
        return self.chunks.pop() if self.chunks else None


@unittest.skipUnless(PYQT_AVAILABLE and SOUNDFILE_AVAILABLE,
                     "PyQt6 and soundfile are required")
class TestAudioVisualizerAppSources(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.path = os.path.join(self.tmpdir, "tone.wav")
        t = np.arange(FILE_FRAMES) / FILE_RATE
        soundfile.write(self.path, 0.5 * np.sin(2 * np.pi * 440 * t), FILE_RATE)

        self.speaker = FakeSpeaker(gated=True)

        def make_player(**kwargs):
            return AudioFilePlayer(speaker_factory=lambda: self.speaker,
                                   **kwargs)

        # QApplication(sys.argv) would raise with the module-level one already
        # up, and the render timer must not fire behind the assertions.
        patches = [
            mock.patch.object(main, "QApplication",
                              return_value=QApplication.instance()),
            mock.patch.object(main, "AudioCaptureSubsystem", FakeCapture),
            mock.patch.object(main, "AudioFilePlayer", side_effect=make_player),
            mock.patch.object(main, "QTimer"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

        self.app = main.AudioVisualizerApp()
        self.addCleanup(self.app.file_player.unload)
        self.addCleanup(lambda: self.speaker.release(chunks=64))
        self.capture = self.app.audio_capture

    # ── Opening a file ──

    def test_opening_a_file_switches_the_source_and_retunes_the_analysis(self):
        self.app.open_file(self.path)

        self.assertEqual(self.app.source, main.SOURCE_FILE)
        self.assertTrue(self.app.file_player.is_playing)
        self.assertEqual(self.app.processor.sample_rate, FILE_RATE)
        # The capture is stopped: a loopback device would otherwise re-capture
        # this very playback on top of the file's own samples.
        self.assertEqual(self.capture.stopped, 1)

    def test_opening_a_file_shows_it_in_the_transport(self):
        self.app.open_file(self.path)

        bar = self.app.main_window.playback_bar
        self.assertTrue(bar.isVisibleTo(self.app.main_window))
        self.assertEqual(bar.track_label.toolTip(), "tone.wav")
        self.assertEqual(bar.duration_label.text(), "0:00")
        self.assertEqual(bar.seek_slider.maximum(),
                         int(FILE_FRAMES / FILE_RATE * 1000))

    def test_an_unreadable_file_reports_an_error_and_keeps_system_audio(self):
        bad = os.path.join(self.tmpdir, "broken.mp3")
        with open(bad, "wb") as handle:
            handle.write(b"not audio")

        self.app.open_file(bad)

        self.assertEqual(self.app.source, main.SOURCE_SYSTEM)
        self.assertEqual(self.capture.stopped, 0)
        status = self.app.main_window.status_label
        self.assertTrue(status.isVisibleTo(self.app.main_window))
        self.assertIn("broken.mp3", status.text())
        self.assertFalse(
            self.app.main_window.playback_bar.isVisibleTo(self.app.main_window))

    # ── Returning to system audio ──

    def test_closing_the_file_restores_system_audio(self):
        self.app.open_file(self.path)
        self.app.close_file()

        self.assertEqual(self.app.source, main.SOURCE_SYSTEM)
        self.assertFalse(self.app.file_player.has_file)
        self.assertEqual(self.app.processor.sample_rate, main.SAMPLE_RATE)
        self.assertEqual(self.capture.started, 1)
        self.assertFalse(
            self.app.main_window.playback_bar.isVisibleTo(self.app.main_window))

    # ── Transport ──

    def test_play_pause_toggles_playback_and_the_button(self):
        self.app.open_file(self.path)
        button = self.app.main_window.playback_bar.play_button

        self.app.toggle_playback()
        self.assertFalse(self.app.file_player.is_playing)
        self.assertEqual(button.accessibleName(), "Play")

        self.app.toggle_playback()
        self.assertTrue(self.app.file_player.is_playing)
        self.assertEqual(button.accessibleName(), "Pause")

    def test_seeking_moves_playback_and_the_slider(self):
        self.app.open_file(self.path)
        self.app.toggle_playback()  # pause, so the position stays put

        self.app.seek(0.2)

        self.assertAlmostEqual(self.app.file_player.position, 0.2, places=3)
        # Frame-quantized, so the slider lands within a millisecond of the ask.
        self.assertAlmostEqual(
            self.app.main_window.playback_bar.seek_slider.value(), 200, delta=1)

    def test_transport_does_nothing_without_a_file(self):
        self.app.toggle_playback()
        self.app.seek(5.0)
        self.assertEqual(self.app.source, main.SOURCE_SYSTEM)
        self.assertFalse(self.app.file_player.has_file)

    # ── The render loop ──

    def test_the_visualizer_reads_the_file_while_one_is_open(self):
        self.app.open_file(self.path)
        self.capture.chunks.append(np.ones(2048))  # must be ignored
        self.speaker.release()

        self.assertTrue(wait_until(
            lambda: len(self.app.file_player.audio_queue) > 0))
        published = self.app.file_player.audio_queue[-1]

        np.testing.assert_array_equal(self.app._poll_source(), published)
        self.assertEqual(len(self.capture.chunks), 1)

    def test_the_visualizer_reads_the_capture_without_a_file(self):
        chunk = np.ones(2048)
        self.capture.chunks.append(chunk)
        np.testing.assert_array_equal(self.app._poll_source(), chunk)

    def test_bars_are_driven_by_the_file(self):
        self.app.open_file(self.path)
        self.speaker.release()
        self.assertTrue(wait_until(
            lambda: len(self.app.file_player.audio_queue) > 0))

        self.app.update_visualizer()

        self.assertTrue(self.app.main_window.bar_widget.bar_values.any(),
                        "the file's audio never reached the bars")

    def test_playback_position_reaches_the_transport(self):
        self.app.open_file(self.path)
        self.app.toggle_playback()
        self.app.file_player.seek(0.15)

        self.app.update_visualizer()

        self.assertAlmostEqual(
            self.app.main_window.playback_bar.seek_slider.value(), 150, delta=1)

    def test_a_playback_failure_is_surfaced(self):
        speaker = FakeSpeaker(play_error=RuntimeError("device disappeared"))
        with mock.patch.object(self.app.file_player, "_speaker_factory",
                               lambda: speaker):
            self.app.open_file(self.path)
            self.assertTrue(wait_until(
                lambda: self.app.file_player.last_error is not None))
            self.app.update_visualizer()

        status = self.app.main_window.status_label
        self.assertTrue(status.isVisibleTo(self.app.main_window))
        self.assertIn("device disappeared", status.text())

    def test_a_capture_failure_is_still_surfaced(self):
        self.capture.last_error = "No loopback-capable microphone found."
        self.app.update_visualizer()

        status = self.app.main_window.status_label
        self.assertTrue(status.isVisibleTo(self.app.main_window))
        self.assertIn("loopback", status.text())


if __name__ == "__main__":
    unittest.main()
