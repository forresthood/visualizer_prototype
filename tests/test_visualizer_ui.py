"""
Unit tests for the visualizer UI.

These construct the real MainWindow. The application previously shipped a
_build_ui that raised NameError before the window was ever built, and nothing
in the suite noticed because no test instantiated a widget.
"""
import os
import unittest
from unittest import mock

import numpy as np

# Must be set before PyQt6 is imported: without it Qt tries the xcb plugin and
# aborts the whole process on a headless machine. conftest.py sets this too,
# but unittest discovery does not read conftest.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QColor
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without PyQt6
    PYQT_AVAILABLE = False

if PYQT_AVAILABLE:
    from PyQt6.QtCore import Qt
    from visualizer_ui import (MainWindow, SpectrogramWidget, WaveformWidget,
                               BarVisualizerWidget, PlaybackBar,
                               format_duration)

_app = None


def setUpModule():
    """A single QApplication is required before any widget is created."""
    global _app
    if PYQT_AVAILABLE:
        _app = QApplication.instance() or QApplication([])


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is not installed")
class TestMainWindow(unittest.TestCase):
    """Test suite for MainWindow construction and its control wiring."""

    def setUp(self):
        self.window = MainWindow()
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()

    def _descends_from_window(self, widget):
        parent = widget
        while parent is not None:
            if parent is self.window:
                return True
            parent = parent.parentWidget()
        return False

    def test_window_constructs(self):
        """MainWindow can be built at all (regression: NameError in _build_ui)."""
        self.assertEqual(self.window.stack.count(), 3)
        self.assertEqual(self.window.current_mode, "Bars")

    def test_all_controls_are_in_the_layout(self):
        """
        Every control must be parented into the window.

        Catches controls that are created and populated but never added to a
        layout, which leaves them invisible rather than raising.
        """
        controls = ([self.window.stack, self.window.bar_slider,
                     self.window.bar_count_label, self.window.bar_count_container,
                     self.window.status_label, self.window.title_bar]
                    + self.window.mode_buttons + self.window.swatches)
        for control in controls:
            with self.subTest(control=control.__class__.__name__):
                self.assertTrue(self._descends_from_window(control),
                                f"{control} is not parented into the window")

    def test_default_color_is_applied_to_widgets(self):
        """The swatch marked active must match the color actually rendered."""
        active = [s.swatch_name for s in self.window.swatches if s.active]
        self.assertEqual(active, [self.window.DEFAULT_COLOR_NAME])

        expected = self.window.color_map[self.window.DEFAULT_COLOR_NAME]
        for widget in (self.window.bar_widget, self.window.waveform_widget,
                       self.window.spectrogram_widget):
            with self.subTest(widget=widget.__class__.__name__):
                self.assertEqual(widget.bar_color, expected)
                self.assertFalse(widget.rainbow_mode)

    def test_mode_buttons_switch_the_stack_and_notify(self):
        """Clicking a mode button must change the view and fire the callback."""
        seen = []
        self.window.on_mode_changed_callback = seen.append

        for button, (_label, mode, index) in zip(self.window.mode_buttons,
                                                 self.window.MODES):
            button.click()
            self.assertEqual(self.window.current_mode, mode)
            self.assertEqual(self.window.stack.currentIndex(), index)
            self.assertEqual([b.mode for b in self.window.mode_buttons
                              if b.isChecked()], [mode])

        self.assertEqual(seen, ["Bars", "Waveform", "Spectrogram"])

    def test_bar_count_visible_only_in_bars_mode(self):
        self.window.mode_buttons[0].click()
        self.assertTrue(self.window.bar_count_container.isVisible())
        self.window.mode_buttons[2].click()
        self.assertFalse(self.window.bar_count_container.isVisible())

    def test_bar_count_snapping_stays_consistent(self):
        """
        Slider, readout, visualizer and callback must all agree.

        If the reported count and the widget's bar count diverge,
        BarVisualizerWidget.update_values silently drops every frame.
        """
        reported = []
        self.window.on_bars_changed_callback = reported.append

        for value, expected in ((33, 32), (100, 96), (9, 8), (128, 128)):
            with self.subTest(value=value):
                self.window.bar_slider.setValue(value)
                self.assertEqual(reported[-1], expected)
                self.assertEqual(self.window.bar_slider.value(), expected)
                self.assertEqual(self.window.bar_count_label.text(), str(expected))
                self.assertEqual(self.window.bar_widget.bars, expected)

                # The count the app would use must be accepted by the widget.
                values = np.zeros(reported[-1])
                self.window.bar_widget.update_values(values)
                self.assertEqual(len(self.window.bar_widget.bar_values), expected)

    def test_color_swatch_click_applies_color(self):
        for name in ("Red", "Green", "Rainbow"):
            with self.subTest(color=name):
                swatch = next(s for s in self.window.swatches
                              if s.swatch_name == name)
                swatch.click()
                self.assertEqual(self.window._current_color_name, name)
                self.assertEqual([s.swatch_name for s in self.window.swatches
                                  if s.active], [name])
                for widget in (self.window.bar_widget,
                               self.window.waveform_widget,
                               self.window.spectrogram_widget):
                    if name == "Rainbow":
                        self.assertTrue(widget.rainbow_mode)
                    else:
                        self.assertFalse(widget.rainbow_mode)
                        self.assertEqual(widget.bar_color,
                                         self.window.color_map[name])

    def test_status_label_reports_errors(self):
        self.window.set_status("capture died", is_error=True)
        self.assertTrue(self.window.status_label.isVisible())
        self.assertIn("capture died", self.window.status_label.text())
        self.window.set_status("")
        self.assertFalse(self.window.status_label.isVisible())


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is not installed")
class TestRainbowTimers(unittest.TestCase):
    """The animation timer must only run when it can be seen."""

    def test_timer_idle_until_rainbow_enabled(self):
        widget = BarVisualizerWidget()
        widget.show()
        self.addCleanup(widget.deleteLater)

        self.assertFalse(widget.rainbow_timer.isActive())
        widget.set_rainbow_mode(True)
        self.assertTrue(widget.rainbow_timer.isActive())
        widget.set_color(QColor(255, 0, 0))
        self.assertFalse(widget.rainbow_timer.isActive())

    def test_timer_stops_while_hidden(self):
        widget = BarVisualizerWidget()
        widget.show()
        self.addCleanup(widget.deleteLater)
        widget.set_rainbow_mode(True)
        self.assertTrue(widget.rainbow_timer.isActive())

        widget.hide()
        self.assertFalse(widget.rainbow_timer.isActive())
        widget.show()
        self.assertTrue(widget.rainbow_timer.isActive())

    def test_only_the_visible_stack_widget_animates(self):
        window = MainWindow()
        window.show()
        self.addCleanup(window.deleteLater)
        next(s for s in window.swatches if s.swatch_name == "Rainbow").click()

        active = [w.__class__.__name__
                  for w in (window.bar_widget, window.waveform_widget,
                            window.spectrogram_widget)
                  if w.rainbow_timer.isActive()]
        self.assertEqual(active, ["BarVisualizerWidget"])


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is not installed")
class TestSpectrogramWidget(unittest.TestCase):
    """The spectrogram must show the whole spectrum, not just what fits."""

    # main.py runs buffer_frames=2048, so get_raw_fft returns 1025 bins.
    NUM_BINS = 1025

    def test_every_bin_is_rendered(self):
        """
        No FFT bin may be dropped.

        Previously bin_height was floored at 1px without rescaling, so bins
        beyond the widget height were drawn at negative y and never seen.
        """
        for bin_index in (0, 1, 400, 512, 900, self.NUM_BINS - 1):
            with self.subTest(bin_index=bin_index):
                widget = SpectrogramWidget()
                widget.resize(850, 400)
                data = np.zeros(self.NUM_BINS)
                data[bin_index] = 1.0
                widget.update_fft(data)
                self.assertTrue(widget._intensity[:, -1].any(),
                                f"bin {bin_index} rendered nowhere")

    def test_low_bins_render_low_and_high_bins_render_high(self):
        """Row 0 is the top of the widget, so frequency must increase upward."""
        rows = []
        for bin_index in (0, self.NUM_BINS - 1):
            widget = SpectrogramWidget()
            data = np.zeros(self.NUM_BINS)
            data[bin_index] = 1.0
            widget.update_fft(data)
            rows.append(int(np.nonzero(widget._intensity[:, -1])[0][0]))

        low_freq_row, high_freq_row = rows
        self.assertGreater(low_freq_row, high_freq_row)

    def test_history_scrolls_left(self):
        widget = SpectrogramWidget(history_length=8)
        loud = np.zeros(self.NUM_BINS)
        loud[10] = 1.0
        widget.update_fft(loud)
        newest = widget._intensity[:, -1].copy()
        self.assertTrue(newest.any())

        widget.update_fft(np.zeros(self.NUM_BINS))
        np.testing.assert_array_equal(widget._intensity[:, -2], newest)
        self.assertFalse(widget._intensity[:, -1].any())

    def test_empty_fft_is_ignored(self):
        widget = SpectrogramWidget()
        widget.update_fft(np.array([]))
        self.assertFalse(widget._intensity.any())


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is not installed")
class TestWaveformWidget(unittest.TestCase):
    """Waveform decimation must stay bounded and preserve peaks."""

    def test_points_are_bounded_by_width(self):
        widget = WaveformWidget()
        xs, _ys = widget._plot_points(np.zeros(8192), 400, 200.0)
        self.assertLessEqual(len(xs), 400)

    def test_short_buffers_are_not_decimated(self):
        widget = WaveformWidget()
        xs, _ys = widget._plot_points(np.zeros(100), 400, 200.0)
        self.assertEqual(len(xs), 100)

    def test_peaks_survive_decimation(self):
        widget = WaveformWidget()
        data = np.zeros(4000)
        data[1234] = 0.9
        mid_y = 200.0
        _xs, ys = widget._plot_points(data, 400, mid_y)
        # The spike must still pull a point away from the center line.
        self.assertAlmostEqual(min(ys), mid_y - 0.9 * mid_y * 0.9, places=6)

    def test_plot_points_returns_plain_floats(self):
        """
        QPainterPath is built from these in a per-point loop; numpy scalars
        make that loop ~1.8x slower for identical geometry.
        """
        widget = WaveformWidget()
        xs, ys = widget._plot_points(np.zeros(4000), 400, 200.0)
        self.assertIsInstance(xs, list)
        self.assertIsInstance(ys, list)
        self.assertIsInstance(xs[0], float)
        self.assertIsInstance(ys[0], float)


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is not installed")
class TestFormatDuration(unittest.TestCase):
    """Track times shown next to the seek slider."""

    def test_formats_minutes_and_seconds(self):
        self.assertEqual(format_duration(0), "0:00")
        self.assertEqual(format_duration(9), "0:09")
        self.assertEqual(format_duration(65), "1:05")
        self.assertEqual(format_duration(599.9), "9:59")

    def test_formats_hours_once_past_the_hour(self):
        self.assertEqual(format_duration(3600), "1:00:00")
        self.assertEqual(format_duration(3725), "1:02:05")

    def test_nonsense_durations_do_not_raise(self):
        for value in (None, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertEqual(format_duration(value), "0:00")


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is not installed")
class TestPlaybackBar(unittest.TestCase):
    """The file transport: visibility, seeking and the play/pause state."""

    def setUp(self):
        self.bar = PlaybackBar()
        self.seeks = []
        self.bar.on_seek = self.seeks.append
        self.play_pauses = []
        self.bar.on_play_pause = lambda: self.play_pauses.append(True)
        self.closes = []
        self.bar.on_close = lambda: self.closes.append(True)

    def test_hidden_until_a_file_is_loaded(self):
        self.assertFalse(self.bar.isVisible())

    def test_set_track_shows_the_bar_and_its_times(self):
        self.bar.set_track("song.flac", 125.0)

        self.assertTrue(self.bar.isVisible())
        self.assertEqual(self.bar.duration_label.text(), "2:05")
        self.assertEqual(self.bar.position_label.text(), "0:00")
        self.assertEqual(self.bar.seek_slider.maximum(), 125000)
        self.assertEqual(self.bar.seek_slider.value(), 0)

    def test_long_names_are_elided_but_kept_in_the_tooltip(self):
        name = "A Really Quite Excessively Long Track Name Indeed.flac"
        self.bar.set_track(name, 10.0)

        self.assertEqual(self.bar.track_label.toolTip(), name)
        self.assertLess(len(self.bar.track_label.text()), len(name))

    def test_clear_track_hides_the_bar(self):
        self.bar.set_track("song.flac", 10.0)
        self.bar.clear_track()
        self.assertFalse(self.bar.isVisible())

    def test_position_moves_the_slider_without_seeking(self):
        """App-driven position updates must not echo back as seek requests."""
        self.bar.set_track("song.flac", 100.0)
        self.bar.set_position(42.0)

        self.assertEqual(self.bar.seek_slider.value(), 42000)
        self.assertEqual(self.bar.position_label.text(), "0:42")
        self.assertEqual(self.seeks, [])

    def test_dragging_the_slider_seeks_on_release(self):
        self.bar.set_track("song.flac", 100.0)

        self.bar.seek_slider.sliderPressed.emit()
        self.bar.seek_slider.setValue(30000)
        self.bar.seek_slider.sliderMoved.emit(30000)
        # Mid-drag the label previews the target but no seek has happened yet.
        self.assertEqual(self.bar.position_label.text(), "0:30")
        self.assertEqual(self.seeks, [])

        self.bar.seek_slider.sliderReleased.emit()
        self.assertEqual(self.seeks, [30.0])

    def test_playback_position_is_ignored_while_scrubbing(self):
        self.bar.set_track("song.flac", 100.0)
        self.bar.seek_slider.sliderPressed.emit()
        self.bar.seek_slider.setValue(30000)

        self.bar.set_position(5.0)

        self.assertEqual(self.bar.seek_slider.value(), 30000)

    def test_keyboard_slider_changes_seek(self):
        """Page/arrow keys never emit sliderPressed, so they seek directly."""
        self.bar.set_track("song.flac", 100.0)
        self.bar.seek_slider.setValue(12000)
        self.assertEqual(self.seeks, [12.0])

    def test_play_button_reports_intent(self):
        self.bar.play_button.click()
        self.assertEqual(len(self.play_pauses), 1)

    def test_close_button_reports_intent(self):
        self.bar.close_button.click()
        self.assertEqual(len(self.closes), 1)

    def test_play_button_reflects_playback_state(self):
        self.bar.set_playing(True)
        self.assertEqual(self.bar.play_button.accessibleName(), "Pause")
        self.bar.set_playing(False)
        self.assertEqual(self.bar.play_button.accessibleName(), "Play")


@unittest.skipUnless(PYQT_AVAILABLE, "PyQt6 is not installed")
class TestMainWindowPlayback(unittest.TestCase):
    """MainWindow's side of file playback."""

    def setUp(self):
        self.window = MainWindow()

    def shown(self):
        """
        Whether the transport would be on screen.

        The window itself is never shown in these tests, and isVisible() is
        False for every child of a hidden window; isVisibleTo answers the
        question we actually care about.
        """
        return self.window.playback_bar.isVisibleTo(self.window)

    def test_transport_is_hidden_until_a_file_is_opened(self):
        self.assertFalse(self.shown())

    def test_opening_a_file_shows_the_transport_playing(self):
        self.window.set_playback_file("song.mp3", 90.0)

        self.assertTrue(self.shown())
        self.assertEqual(self.window.playback_bar.duration_label.text(), "1:30")
        self.assertEqual(self.window.playback_bar.play_button.accessibleName(),
                         "Pause")

    def test_clearing_the_file_hides_the_transport(self):
        self.window.set_playback_file("song.mp3", 90.0)
        self.window.clear_playback_file()
        self.assertFalse(self.shown())

    def test_open_button_prompts_for_a_file_and_reports_the_choice(self):
        opened = []
        self.window.on_file_opened_callback = opened.append

        with mock.patch("visualizer_ui.QFileDialog.getOpenFileName",
                        return_value=("/music/song.flac", "")):
            self.window.open_audio_file()

        self.assertEqual(opened, ["/music/song.flac"])
        # The next dialog reopens where the last one left off.
        self.assertEqual(self.window._last_open_dir, "/music")

    def test_cancelling_the_dialog_opens_nothing(self):
        opened = []
        self.window.on_file_opened_callback = opened.append

        with mock.patch("visualizer_ui.QFileDialog.getOpenFileName",
                        return_value=("", "")):
            self.window.open_audio_file()

        self.assertEqual(opened, [])

    def test_transport_interactions_reach_the_application(self):
        events = []
        self.window.on_play_pause_callback = lambda: events.append("play_pause")
        self.window.on_seek_callback = lambda s: events.append(("seek", s))
        self.window.on_close_file_callback = lambda: events.append("close")

        self.window.playback_bar.play_button.click()
        self.window.playback_bar.set_track("song.mp3", 100.0)
        self.window.playback_bar.seek_slider.sliderPressed.emit()
        self.window.playback_bar.seek_slider.setValue(25000)
        self.window.playback_bar.seek_slider.sliderReleased.emit()
        self.window.playback_bar.close_button.click()

        self.assertEqual(events, ["play_pause", ("seek", 25.0), "close"])

    def test_transport_callbacks_are_optional(self):
        """An unwired window must not raise when its transport is used."""
        self.window.playback_bar.play_button.click()
        self.window.playback_bar.close_button.click()
        with mock.patch("visualizer_ui.QFileDialog.getOpenFileName",
                        return_value=("/music/song.flac", "")):
            self.window.open_audio_file()

    def test_space_only_takes_over_while_a_file_is_open(self):
        """
        A window-scoped Space shortcut would otherwise steal the key from
        whichever control has focus.
        """
        self.assertFalse(self.window.play_pause_shortcut.isEnabled())

        self.window.set_playback_file("song.mp3", 90.0)
        self.assertTrue(self.window.play_pause_shortcut.isEnabled())

        self.window.clear_playback_file()
        self.assertFalse(self.window.play_pause_shortcut.isEnabled())

    def test_open_file_button_lives_in_the_title_bar(self):
        """The controls row is already tight at the minimum window width."""
        opened = []
        self.window.on_file_opened_callback = opened.append

        with mock.patch("visualizer_ui.QFileDialog.getOpenFileName",
                        return_value=("/music/song.flac", "")):
            self.window.title_bar.open_file_button.click()

        self.assertEqual(opened, ["/music/song.flac"])


if __name__ == '__main__':
    unittest.main()
