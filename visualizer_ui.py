import sys
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QComboBox, 
                             QPushButton, QSpinBox, QGroupBox, QStackedWidget)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QLinearGradient,
                          QImage, QPainterPath)

# ─────────────────────────────────────────────
#  Base mixin for shared color/rainbow logic
# ─────────────────────────────────────────────
class ColorMixin:
    """Shared color and rainbow state for all visualizer widgets."""
    def init_color(self):
        self.bar_color = QColor(0, 200, 255)
        self.bg_color = QColor(20, 20, 20)
        self.rainbow_mode = False
        self.rainbow_hue = 0.0

        self.rainbow_timer = QTimer(self)
        self.rainbow_timer.timeout.connect(self._update_rainbow)
        self.rainbow_timer.start(16)

    def set_color(self, color):
        self.rainbow_mode = False
        self.bar_color = color
        self.update()

    def set_rainbow_mode(self, enabled):
        self.rainbow_mode = enabled

    def _update_rainbow(self):
        if self.rainbow_mode:
            self.rainbow_hue += 0.005
            if self.rainbow_hue > 1.0:
                self.rainbow_hue -= 1.0
            self.bar_color = QColor.fromHsvF(self.rainbow_hue, 0.8, 1.0)
            self.update()

# ─────────────────────────────────────────────
#  1. Bar Visualizer (existing, refactored)
# ─────────────────────────────────────────────
class BarVisualizerWidget(ColorMixin, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_color()
        self.bars = 32
        self.bar_values = np.zeros(self.bars)

    def set_bars(self, count):
        self.bars = count
        self.bar_values = np.zeros(self.bars)
        self.update()

    def update_values(self, new_values):
        if len(new_values) != self.bars:
            return
        decay_rate = 0.85
        self.bar_values = np.maximum(self.bar_values * decay_rate, new_values)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)

        w, h = self.width(), self.height()
        spacing = max(1, w // (self.bars * 4))
        total_spacing = spacing * (self.bars + 1)
        bar_width = (w - total_spacing) / self.bars
        if bar_width <= 0:
            return

        for i in range(self.bars):
            val = max(0.0, min(1.0, self.bar_values[i]))
            bar_height = val * (h - 20)
            x = spacing + i * (bar_width + spacing)
            y = h - bar_height - 10
            rect = QRectF(x, y, bar_width, bar_height)

            if self.rainbow_mode:
                current_color = QColor.fromHsvF((self.rainbow_hue + i / self.bars) % 1.0, 0.8, 1.0)
            else:
                current_color = self.bar_color

            gradient = QLinearGradient(x, y + bar_height, x, y)
            gradient.setColorAt(0, current_color.darker(150))
            gradient.setColorAt(1, current_color.lighter(120))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(rect, 4, 4)

# ─────────────────────────────────────────────
#  2. Waveform Visualizer
# ─────────────────────────────────────────────
class WaveformWidget(ColorMixin, QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_color()
        self.waveform_data = np.zeros(2048)

    def update_waveform(self, audio_data):
        self.waveform_data = audio_data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)

        w, h = self.width(), self.height()
        mid_y = h / 2.0
        n = len(self.waveform_data)
        if n == 0:
            return

        # Build the waveform path
        path = QPainterPath()
        step = w / n

        for i in range(n):
            x = i * step
            # Scale sample value to fill half the widget height
            y = mid_y - self.waveform_data[i] * mid_y * 0.9
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        # Draw the waveform line
        if self.rainbow_mode:
            # Draw a gradient-colored line by using multiple short segments
            segment_count = min(n, 200)
            segment_size = max(1, n // segment_count)
            for seg in range(0, n, segment_size):
                seg_path = QPainterPath()
                hue = (self.rainbow_hue + seg / n) % 1.0
                pen = QPen(QColor.fromHsvF(hue, 0.8, 1.0), 2)
                for j in range(seg, min(seg + segment_size + 1, n)):
                    x = j * step
                    y = mid_y - self.waveform_data[j] * mid_y * 0.9
                    if j == seg:
                        seg_path.moveTo(x, y)
                    else:
                        seg_path.lineTo(x, y)
                painter.setPen(pen)
                painter.drawPath(seg_path)
        else:
            pen = QPen(self.bar_color, 2)
            painter.setPen(pen)
            painter.drawPath(path)

        # Draw center line
        center_pen = QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine)
        painter.setPen(center_pen)
        painter.drawLine(0, int(mid_y), w, int(mid_y))

# ─────────────────────────────────────────────
#  3. Spectrogram Visualizer
# ─────────────────────────────────────────────
class SpectrogramWidget(ColorMixin, QWidget):
    def __init__(self, parent=None, history_length=200):
        super().__init__(parent)
        self.init_color()
        self.history_length = history_length
        self.fft_history = deque(maxlen=history_length)

    def update_fft(self, fft_data):
        self.fft_history.append(fft_data)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.bg_color)

        w, h = self.width(), self.height()
        num_cols = len(self.fft_history)
        if num_cols == 0:
            return

        # Each column is one time slice; we stretch to fill width
        col_width = max(1.0, w / self.history_length)

        for col_idx, fft_slice in enumerate(self.fft_history):
            x = int(col_idx * col_width)
            num_bins = len(fft_slice)
            if num_bins == 0:
                continue

            bin_height = max(1.0, h / num_bins)

            for bin_idx in range(num_bins):
                intensity = max(0.0, min(1.0, fft_slice[bin_idx]))
                if intensity < 0.01:
                    continue

                # Frequency axis: low freqs at bottom, high at top
                y = int(h - (bin_idx + 1) * bin_height)

                if self.rainbow_mode:
                    hue = (self.rainbow_hue + bin_idx / num_bins) % 1.0
                    color = QColor.fromHsvF(hue, 0.8, intensity)
                else:
                    color = QColor.fromHsvF(
                        self.bar_color.hsvHueF() if self.bar_color.hsvHueF() >= 0 else 0.5,
                        0.8,
                        intensity
                    )

                painter.fillRect(QRectF(x, y, col_width + 1, bin_height + 1), color)

# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Audio Visualizer")
        self.resize(800, 600)

        self.bar_widget = BarVisualizerWidget()
        self.waveform_widget = WaveformWidget()
        self.spectrogram_widget = SpectrogramWidget()

        # Keep a reference for backward compat (main.py uses self.visualizer)
        self.visualizer = self.bar_widget
        self.current_mode = "Bars"

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Stacked widget for swapping visualizations
        self.stack = QStackedWidget()
        self.stack.addWidget(self.bar_widget)        # index 0
        self.stack.addWidget(self.waveform_widget)    # index 1
        self.stack.addWidget(self.spectrogram_widget) # index 2
        main_layout.addWidget(self.stack, stretch=1)

        # Controls Group
        controls_group = QGroupBox("Settings")
        controls_layout = QHBoxLayout(controls_group)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_label = QLabel("&Mode:")
        mode_layout.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.setToolTip("Select the visualization type (Alt+M)")
        mode_label.setBuddy(self.mode_combo)
        self.mode_combo.addItems(["Bars", "Waveform", "Spectrogram"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        controls_layout.addLayout(mode_layout)

        # Bar Count Control
        self.bar_count_layout_widget = QWidget()
        bar_count_layout = QHBoxLayout(self.bar_count_layout_widget)
        bar_count_layout.setContentsMargins(0, 0, 0, 0)
        bar_label = QLabel("&Bars:")
        bar_count_layout.addWidget(bar_label)
        self.bar_spinbox = QSpinBox()
        self.bar_spinbox.setToolTip("Adjust the number of frequency bars (Alt+B)")
        bar_label.setBuddy(self.bar_spinbox)
        self.bar_spinbox.setRange(8, 256)
        self.bar_spinbox.setValue(32)
        self.bar_spinbox.setSingleStep(8)
        self.bar_spinbox.valueChanged.connect(self._on_bars_changed)
        bar_count_layout.addWidget(self.bar_spinbox)
        controls_layout.addWidget(self.bar_count_layout_widget)

        # Color Combo Box
        color_layout = QHBoxLayout()
        color_label = QLabel("&Color:")
        color_layout.addWidget(color_label)
        self.color_combo = QComboBox()
        self.color_combo.setToolTip("Choose the color theme (Alt+C)")
        color_label.setBuddy(self.color_combo)

        self.color_map = {
            "Rainbow": None,
            "Cyan": QColor(0, 200, 255),
            "Red": QColor(255, 50, 50),
            "Orange": QColor(255, 128, 0),
            "Yellow": QColor(255, 255, 0),
            "Green": QColor(50, 255, 50),
            "Blue": QColor(50, 50, 255),
            "Purple": QColor(150, 50, 255),
            "Magenta": QColor(255, 0, 255)
        }

        self.color_combo.addItems(list(self.color_map.keys()))
        self.color_combo.setCurrentIndex(1)  # Cyan default
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        color_layout.addWidget(self.color_combo)
        controls_layout.addLayout(color_layout)

        main_layout.addWidget(controls_group)

    # ── Helpers ──

    def _active_widget(self):
        return self.stack.currentWidget()

    def _on_mode_changed(self, mode_name):
        self.current_mode = mode_name
        index_map = {"Bars": 0, "Waveform": 1, "Spectrogram": 2}
        self.stack.setCurrentIndex(index_map.get(mode_name, 0))

        # Show bar count spinner only in Bars mode
        self.bar_count_layout_widget.setVisible(mode_name == "Bars")

        # Notify main.py that mode has changed
        if hasattr(self, "on_mode_changed_callback"):
            self.on_mode_changed_callback(mode_name)

        # Re-apply current color to the new active widget
        self._on_color_changed(self.color_combo.currentText())

    def _on_bars_changed(self, value):
        self.bar_widget.set_bars(value)
        if hasattr(self, "on_bars_changed_callback"):
            self.on_bars_changed_callback(value)

    def _on_color_changed(self, color_name):
        color = self.color_map.get(color_name)
        # Apply to ALL widgets so switching modes preserves the color
        for widget in [self.bar_widget, self.waveform_widget, self.spectrogram_widget]:
            if color is None:
                widget.set_rainbow_mode(True)
            else:
                widget.set_color(color)
