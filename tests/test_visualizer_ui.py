"""
Unit tests for the visualizer UI.

These construct the real MainWindow. The application previously shipped a
_build_ui that raised NameError before the window was ever built, and nothing
in the suite noticed because no test instantiated a widget.
"""
import os
import unittest

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
    from visualizer_ui import (MainWindow, SpectrogramWidget, WaveformWidget,
                               BarVisualizerWidget)

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


if __name__ == '__main__':
    unittest.main()
