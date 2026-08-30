import os

import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider,
                             QStackedWidget, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QLinearGradient,
                         QPainterPath, QRadialGradient, QConicalGradient,
                         QMouseEvent, QFont, QImage, QFontMetrics, QKeySequence,
                         QShortcut)

from audio_file import file_dialog_filter

class Theme:
    VIZ_BG = QColor(13, 13, 22)
    BG_GLASS = QColor(30, 30, 40, 180)
    BG_GLASS_BORDER = QColor(255, 255, 255, 20)

    TEXT_PRIMARY = QColor(240, 240, 245)
    TEXT_SECONDARY = QColor(160, 160, 170)
    TEXT_ERROR = QColor(255, 120, 120)

    SEG_ACTIVE_BG = QColor(255, 255, 255, 20)
    SEG_BORDER = QColor(255, 255, 255, 30)

    RADIUS_LG = 14
    RADIUS_MD = 10
    RADIUS_SM = 6

    FONT_SIZE_TITLE = 14
    FONT_SIZE_SMALL = 11

    @staticmethod
    def font(size, weight=QFont.Weight.Normal):
        f = QFont("Segoe UI", size)
        f.setWeight(weight)
        return f

# Default visualizer color, matching DEFAULT_COLOR_NAME in MainWindow.
DEFAULT_COLOR = QColor(0, 200, 255)

# Shared by every horizontal slider in the controls (bar count, seek), so they
# read as one control set instead of drifting apart.
SLIDER_STYLE = """
    QSlider::groove:horizontal {
        background: rgba(255,255,255,0.08);
        height: 4px;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #7a9aff, stop:1 #5882ff);
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
        border: 1px solid rgba(255,255,255,0.15);
    }
    QSlider::handle:horizontal:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #8eabff, stop:1 #6d95ff);
    }
    QSlider::sub-page:horizontal {
        background: rgba(88,130,255,0.35);
        border-radius: 2px;
    }
"""

# Small pill-shaped button used for the transport controls and the file picker.
TOOL_BUTTON_STYLE = """
    QPushButton {
        border: none;
        border-radius: %(radius)dpx;
        background: rgba(255,255,255,0.06);
        color: rgba(240,240,245,0.85);
        font-size: %(font_size)dpx;
        padding: 0px 10px;
    }
    QPushButton:hover {
        background: rgba(255,255,255,0.14);
        color: rgba(240,240,245,1.0);
    }
    QPushButton:pressed {
        background: rgba(255,255,255,0.20);
    }
"""


def tool_button(text, tooltip, accessible_name, size):
    """A compact pill button matching the controls panel styling."""
    button = QPushButton(text)
    button.setFixedSize(*size)
    button.setStyleSheet(TOOL_BUTTON_STYLE % {"radius": Theme.RADIUS_SM,
                                              "font_size": Theme.FONT_SIZE_SMALL})
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setToolTip(tooltip)
    button.setAccessibleName(accessible_name)
    return button


def format_duration(seconds):
    """Format a number of seconds as m:ss, or h:mm:ss past the hour mark."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        seconds = 0
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

# Rainbow animation tick, ~60 FPS. Only runs while rainbow mode is on and the
# owning widget is visible (see ColorMixin._sync_rainbow_timer).
RAINBOW_TICK_MS = 16

# ─────────────────────────────────────────────
#  Shared rainbow palette
# ─────────────────────────────────────────────
# Building QColors is expensive relative to a 60 FPS paint budget, so the
# (base, dark, light) triples are precomputed once per process and shared by
# every visualizer widget rather than rebuilt per instance or per frame.
RAINBOW_PALETTE_SIZE = 1024
_RAINBOW_PALETTE = None


def rainbow_palette():
    """Return the shared rainbow palette, building it on first use."""
    global _RAINBOW_PALETTE
    if _RAINBOW_PALETTE is None:
        palette = []
        for i in range(RAINBOW_PALETTE_SIZE):
            color = QColor.fromHsvF(i / RAINBOW_PALETTE_SIZE, 0.75, 1.0)
            palette.append((color, color.darker(160), color.lighter(130)))
        _RAINBOW_PALETTE = palette
    return _RAINBOW_PALETTE


def rainbow_colors(fraction):
    """(base, dark, light) colors for a hue position in [0, 1)."""
    idx = int((fraction % 1.0) * RAINBOW_PALETTE_SIZE) % RAINBOW_PALETTE_SIZE
    return rainbow_palette()[idx]


# ─────────────────────────────────────────────
#  Base mixin for shared color / rainbow logic
# ─────────────────────────────────────────────
class ColorMixin:
    """Shared color and rainbow state for all visualizer widgets."""

    # Whether a rainbow tick should trigger a repaint. Widgets that bake colors
    # into a cached surface (the spectrogram) opt out.
    REPAINT_ON_RAINBOW_TICK = True

    def init_color(self):
        self.bar_color = QColor(DEFAULT_COLOR)
        self.bg_color = Theme.VIZ_BG
        self.rainbow_mode = False
        self.rainbow_hue = 0.0

        # Left stopped: an idle timer ticking 60 times a second per widget does
        # nothing useful when rainbow mode is off or the widget is off-screen.
        self.rainbow_timer = QTimer(self)
        self.rainbow_timer.timeout.connect(self._update_rainbow)

    def set_color(self, color):
        self.rainbow_mode = False
        self.bar_color = color
        self._sync_rainbow_timer()
        self.on_palette_changed()
        self.update()

    def set_rainbow_mode(self, enabled):
        self.rainbow_mode = enabled
        self._sync_rainbow_timer()
        self.on_palette_changed()
        self.update()

    def on_palette_changed(self):
        """Hook for subclasses that cache rendered colors."""

    def _sync_rainbow_timer(self):
        """Run the animation timer only when it can actually be seen."""
        should_run = self.rainbow_mode and self.isVisible()
        if should_run and not self.rainbow_timer.isActive():
            self.rainbow_timer.start(RAINBOW_TICK_MS)
        elif not should_run and self.rainbow_timer.isActive():
            self.rainbow_timer.stop()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_rainbow_timer()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._sync_rainbow_timer()

    def _update_rainbow(self):
        self.rainbow_hue += 0.005
        if self.rainbow_hue > 1.0:
            self.rainbow_hue -= 1.0
        self.bar_color = rainbow_colors(self.rainbow_hue)[0]
        if self.REPAINT_ON_RAINBOW_TICK:
            self.update()


# ─────────────────────────────────────────────
#  1. Bar Visualizer
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
        padding_bottom = 30
        padding_top = 20
        draw_h = h - padding_bottom - padding_top
        reflect_h = int(draw_h * 0.12)

        spacing = max(2, int(w / (self.bars * 3.5)))
        total_spacing = spacing * (self.bars + 1)
        bar_width = (w - total_spacing) / self.bars
        if bar_width <= 0:
            return

        # Pre-calculate non-rainbow colors if needed
        if not self.rainbow_mode:
            static_color = self.bar_color
            static_dark = static_color.darker(160)
            static_light = static_color.lighter(130)

        for i in range(self.bars):
            val = max(0.0, min(1.0, self.bar_values[i]))
            bar_height = val * draw_h
            x = spacing + i * (bar_width + spacing)
            y = h - padding_bottom - bar_height
            rect = QRectF(x, y, bar_width, bar_height)

            # Determine color
            if self.rainbow_mode:
                current_color, dark_color, light_color = rainbow_colors(
                    self.rainbow_hue + i / self.bars)
            else:
                current_color = static_color
                dark_color = static_dark
                light_color = static_light

            # Glow pass — larger, semi-transparent behind the bar
            if val > 0.05:
                glow_color = QColor(current_color)
                glow_color.setAlpha(int(35 * val))
                glow_rect = QRectF(x - 3, y - 3, bar_width + 6, bar_height + 6)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(glow_color))
                painter.drawRoundedRect(glow_rect, 6, 6)

            # Main bar gradient (bottom dark → top light)
            gradient = QLinearGradient(0, 1, 0, 0)
            gradient.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
            gradient.setColorAt(0, dark_color)
            gradient.setColorAt(0.5, current_color)
            gradient.setColorAt(1, light_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(rect, 4, 4)

            # Reflection pass
            if val > 0.05 and reflect_h > 2:
                ref_top = h - padding_bottom + 4
                ref_height = min(reflect_h, int(bar_height * 0.25))
                if ref_height > 1:
                    ref_grad = QLinearGradient(0, 0, 0, 1)
                    ref_grad.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
                    ref_color = QColor(current_color)
                    ref_color.setAlpha(45)
                    ref_grad.setColorAt(0, ref_color)
                    ref_color.setAlpha(0)
                    ref_grad.setColorAt(1, ref_color)
                    painter.setBrush(QBrush(ref_grad))
                    painter.drawRoundedRect(QRectF(x, ref_top, bar_width, ref_height), 2, 2)

        # Baseline glow line
        baseline_y = h - padding_bottom
        line_grad = QLinearGradient(0, 0, w, 0)
        line_grad.setColorAt(0, QColor(255, 255, 255, 0))
        line_grad.setColorAt(0.2, QColor(255, 255, 255, 18))
        line_grad.setColorAt(0.8, QColor(255, 255, 255, 18))
        line_grad.setColorAt(1, QColor(255, 255, 255, 0))
        painter.setPen(QPen(QBrush(line_grad), 1))
        painter.drawLine(0, baseline_y, w, baseline_y)


# ─────────────────────────────────────────────
#  2. Waveform Visualizer
# ─────────────────────────────────────────────
class WaveformWidget(ColorMixin, QWidget):
    # Number of colored segments the rainbow trace is split into.
    RAINBOW_SEGMENTS = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_color()
        self.waveform_data = np.zeros(2048)

    def update_waveform(self, audio_data):
        self.waveform_data = audio_data
        self.update()

    def _plot_points(self, data, width, mid_y):
        """
        Map samples to screen coordinates, decimating to one point per pixel
        column when there are more samples than the display can resolve.

        Each column keeps its largest-magnitude sample, so peaks survive while
        the vertex count is bounded by widget width rather than buffer size.
        Stroking cost scales with the path's vertex count and arc length, and
        vertices closer together than a pixel cannot be seen anyway.
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        if width > 0 and n > width:
            cols = width
            usable = (n // cols) * cols
            chunks = data[:usable].reshape(cols, -1)
            peak = np.abs(chunks).argmax(axis=1)
            values = chunks[np.arange(cols), peak]
            xs = np.arange(cols) * (width / cols)
        else:
            values = data
            xs = np.arange(n) * (width / n)

        # Hand back plain Python floats: indexing a numpy array in the
        # QPainterPath loop below builds a numpy scalar per point, which costs
        # ~1.8x more than iterating a list for identical geometry.
        return xs.tolist(), (mid_y - values * mid_y * 0.9).tolist()

    @staticmethod
    def _path_from(xs, ys, start=0, stop=None):
        path = QPainterPath()
        stop = len(xs) if stop is None else stop
        if stop <= start:
            return path
        points = zip(xs[start:stop], ys[start:stop])
        x0, y0 = next(points)
        path.moveTo(x0, y0)
        for x, y in points:
            path.lineTo(x, y)
        return path

    def _segment_paths(self, xs, ys):
        """
        Split the trace into colored segments, returning (hue, path) pairs.
        Built once and reused by both the glow and crisp passes.
        """
        n = len(xs)
        segment_size = max(1, n // min(n, self.RAINBOW_SEGMENTS))
        segments = []
        for start in range(0, n, segment_size):
            # +1 so consecutive segments share a point and the trace has no gaps.
            stop = min(start + segment_size + 1, n)
            segments.append(((self.rainbow_hue + start / n) % 1.0,
                             self._path_from(xs, ys, start, stop)))
        return segments

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)

        w, h = self.width(), self.height()
        mid_y = h / 2.0
        n = len(self.waveform_data)
        if n == 0 or w <= 0:
            return

        # Subtle grid lines
        grid_pen = QPen(QColor(255, 255, 255, 12), 1)
        painter.setPen(grid_pen)
        for frac in [0.25, 0.5, 0.75]:
            gy = int(h * frac)
            painter.drawLine(0, gy, w, gy)

        xs, ys = self._plot_points(self.waveform_data, w, mid_y)

        if self.rainbow_mode:
            segments = self._segment_paths(xs, ys)

            # Glow pass for rainbow
            for hue, path in segments:
                glow_color = QColor(rainbow_colors(hue)[0])
                glow_color.setAlpha(64)
                painter.setPen(QPen(glow_color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawPath(path)

            # Crisp pass for rainbow
            for hue, path in segments:
                painter.setPen(QPen(rainbow_colors(hue)[0], 2))
                painter.drawPath(path)
        else:
            path = self._path_from(xs, ys)

            # Glow pass
            glow_color = QColor(self.bar_color)
            glow_color.setAlpha(50)
            painter.setPen(QPen(glow_color, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(path)

            # Crisp pass
            painter.setPen(QPen(self.bar_color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(path)


# Spectrogram color tables depend only on the image row count, so they are
# built once per row count and shared by every spectrogram instance.
_SPECTROGRAM_LUTS = {}


def _spectrogram_luts(widget):
    """(viridis, rainbow) lookup tables for a spectrogram of `widget.rows` rows."""
    luts = _SPECTROGRAM_LUTS.get(widget.rows)
    if luts is None:
        luts = (widget._build_viridis_lut(), widget._build_rainbow_lut())
        _SPECTROGRAM_LUTS[widget.rows] = luts
    return luts


# ─────────────────────────────────────────────
#  3. Spectrogram Visualizer
# ─────────────────────────────────────────────
class SpectrogramWidget(ColorMixin, QWidget):
    """
    Scrolling spectrogram rendered as a single image.

    The history is kept as an off-screen ARGB buffer that is scrolled one
    column per frame; paintEvent blits it in one call and lets Qt scale it to
    the widget. This keeps every FFT bin on screen regardless of widget height
    (bins are max-reduced onto IMAGE_ROWS rows) and replaces the
    columns x bins grid of fillRect calls that could not hold 60 FPS.
    """

    # Vertical resolution of the spectrogram image; scaled to the widget on
    # paint, so it is independent of the window size.
    IMAGE_ROWS = 256

    # Intensities at or below this index render as background, matching the
    # previous "skip near-silent cells" behaviour.
    SILENCE_INDEX = 2

    # Colors are baked into each column as it arrives, so a rainbow history
    # keeps the hue it was drawn with instead of being recolored every tick.
    REPAINT_ON_RAINBOW_TICK = False

    def __init__(self, parent=None, history_length=200):
        super().__init__(parent)
        self.init_color()
        self.history_length = history_length
        self.rows = self.IMAGE_ROWS

        # Intensity indices (0-255) per image pixel, kept so the image can be
        # rebuilt when the palette changes. Column 0 is oldest, row 0 is the
        # top of the widget (highest frequency).
        self._intensity = np.zeros((self.rows, history_length), dtype=np.uint8)
        self._argb = np.zeros((self.rows, history_length), dtype=np.uint32)

        self._viridis_lut, self._rainbow_lut = _spectrogram_luts(self)
        self._row_edge_cache = {}
        self._rebuild_image()

    # ── Color lookup tables ──

    def _viridis_color(self, t):
        """Attempt a perceptually-uniform viridis-style colormap."""
        t = max(0.0, min(1.0, t))
        if t < 0.25:
            s = t / 0.25
            r, g, b = 0.05 + 0.15 * s, 0.0 + 0.15 * s, 0.2 + 0.35 * s
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            r, g, b = 0.2 - 0.1 * s, 0.15 + 0.35 * s, 0.55 + 0.1 * s
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            r, g, b = 0.1 + 0.4 * s, 0.5 + 0.2 * s, 0.65 - 0.2 * s
        else:
            s = (t - 0.75) / 0.25
            r, g, b = 0.5 + 0.5 * s, 0.7 + 0.2 * s, 0.45 - 0.3 * s
        return QColor.fromRgbF(min(r, 1), min(g, 1), min(b, 1))

    @staticmethod
    def _pack(r, g, b):
        """Pack RGB arrays into Format_RGB32 words (0xffRRGGBB)."""
        return (0xFF000000
                | (r.astype(np.uint32) << 16)
                | (g.astype(np.uint32) << 8)
                | b.astype(np.uint32))

    def _silence_argb(self):
        c = self.bg_color
        return 0xFF000000 | (c.red() << 16) | (c.green() << 8) | c.blue()

    def _build_viridis_lut(self):
        colors = [self._viridis_color(i / 255.0) for i in range(256)]
        rgb = np.array([[c.red(), c.green(), c.blue()] for c in colors])
        lut = self._pack(rgb[:, 0], rgb[:, 1], rgb[:, 2])
        lut[:self.SILENCE_INDEX + 1] = self._silence_argb()
        return lut

    def _build_rainbow_lut(self):
        """
        (rows, 256) table of hue-by-row, brightness-by-intensity colors.

        HSV value scales R, G and B linearly, so one fully-bright color per row
        is enough — the 256 intensity levels are a vectorized multiply rather
        than 65,536 QColor constructions.
        """
        # Row 0 is the top of the widget, i.e. the highest frequency.
        hues = [(self.rows - 1 - r) / self.rows for r in range(self.rows)]
        base = np.array([[c.red(), c.green(), c.blue()]
                         for c in (QColor.fromHsvF(h, 0.8, 1.0) for h in hues)],
                        dtype=np.float64)
        levels = np.arange(256) / 255.0
        rgb = base[:, None, :] * levels[None, :, None]
        lut = self._pack(rgb[..., 0], rgb[..., 1], rgb[..., 2])
        lut[:, :self.SILENCE_INDEX + 1] = self._silence_argb()
        return lut

    def _rainbow_row_index(self):
        """Row-to-hue mapping, rotated so the rainbow drifts over time."""
        offset = int(self.rainbow_hue * self.rows) % self.rows
        return (np.arange(self.rows) + offset) % self.rows

    def _column_colors(self, column):
        if self.rainbow_mode:
            return self._rainbow_lut[self._rainbow_row_index(), column]
        return self._viridis_lut[column]

    def _rebuild_image(self):
        if self.rainbow_mode:
            rows_idx = self._rainbow_row_index()
            self._argb[:] = self._rainbow_lut[rows_idx[:, None], self._intensity]
        else:
            self._argb[:] = self._viridis_lut[self._intensity]

    def on_palette_changed(self):
        self._rebuild_image()

    # ── Data ──

    def _row_edges(self, num_bins):
        """
        Cached reduceat edges mapping `num_bins` FFT bins onto `rows` image
        rows. Every bin contributes to exactly one row, so nothing is clipped
        off the top of the widget no matter how many bins there are.
        """
        edges = self._row_edge_cache.get(num_bins)
        if edges is None:
            edges = np.linspace(0, num_bins, self.rows + 1).astype(int)
            # reduceat needs in-range starts; equal consecutive starts (rows
            # finer than the FFT) fall back to that single bin's value.
            edges[:-1] = np.minimum(edges[:-1], num_bins - 1)
            self._row_edge_cache[num_bins] = edges
        return edges

    def update_fft(self, fft_data):
        fft_data = np.asarray(fft_data, dtype=float)
        if fft_data.size == 0:
            return

        edges = self._row_edges(fft_data.size)
        # Max-reduce bins into rows (peaks survive), then flip so row 0 is the
        # highest frequency at the top of the widget.
        row_values = np.maximum.reduceat(fft_data, edges[:-1])[::-1]
        column = (np.clip(row_values, 0.0, 1.0) * 255).astype(np.uint8)

        # Scroll one column left and append the new column on the right.
        self._intensity[:, :-1] = self._intensity[:, 1:]
        self._intensity[:, -1] = column
        self._argb[:, :-1] = self._argb[:, 1:]
        self._argb[:, -1] = self._column_colors(column)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Constructed per paint so it always wraps the current buffer contents.
        image = QImage(self._argb.data, self.history_length, self.rows,
                       self.history_length * 4, QImage.Format.Format_RGB32)
        painter.drawImage(self.rect(), image)


# ─────────────────────────────────────────────
#  Segmented control button
# ─────────────────────────────────────────────
class SegmentButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)
        self.setMinimumWidth(80)
        self.setFont(Theme.font(Theme.FONT_SIZE_SMALL, QFont.Weight.Medium))
        self.setStyleSheet("border: none; background: transparent; color: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        if self.isChecked():
            painter.setBrush(QBrush(Theme.SEG_ACTIVE_BG))
            painter.setPen(QPen(Theme.SEG_BORDER, 1))
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), Theme.RADIUS_SM, Theme.RADIUS_SM)
            painter.setPen(Theme.TEXT_PRIMARY)
        elif self.underMouse():
            painter.setBrush(QBrush(QColor(255, 255, 255, 8)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), Theme.RADIUS_SM, Theme.RADIUS_SM)
            painter.setPen(Theme.TEXT_SECONDARY)
        else:
            painter.setPen(Theme.TEXT_SECONDARY)

        painter.setFont(self.font())
        # Qt keeps the mnemonic marker in text(); strip it for display so the
        # button reads "Bars", not "&Bars".
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                         self.text().replace("&&", "\0").replace("&", "").replace("\0", "&"))


# ─────────────────────────────────────────────
#  Color swatch button
# ─────────────────────────────────────────────
class ColorSwatch(QPushButton):
    def __init__(self, color, name, is_rainbow=False, parent=None):
        super().__init__(parent)
        self.swatch_color = color
        self.swatch_name = name
        self.is_rainbow = is_rainbow
        self.active = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(28, 28)
        self.setToolTip(name)
        self.setAccessibleName(name)
        self.setStyleSheet("border: none; background: transparent;")

    def set_active(self, active):
        self.active = active
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() / 2, self.height() / 2
        radius = 10

        if self.is_rainbow:
            # Rainbow ring as a single stroked ellipse with a conical gradient,
            # rather than 360 individually-drawn points.
            ring = QConicalGradient(QPointF(cx, cy), 0.0)
            for i in range(13):
                stop = i / 12.0
                ring.setColorAt(stop, QColor.fromHsvF(min(stop, 0.999), 0.85, 1.0))
            painter.setPen(QPen(QBrush(ring), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius - 1, radius - 1)
            # Fill center
            grad = QRadialGradient(cx, cy, radius * 0.65)
            grad.setColorAt(0, QColor(255, 255, 255, 80))
            grad.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(QPointF(cx, cy), radius * 0.6, radius * 0.6)
        else:
            # Filled circle
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.swatch_color))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Active ring
        if self.active:
            ring_pen = QPen(QColor(255, 255, 255, 200), 2)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius + 2, radius + 2)

            # Checkmark
            painter.setPen(QPen(QColor(255, 255, 255, 230), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            check_path = QPainterPath()
            check_path.moveTo(cx - 3.5, cy)
            check_path.lineTo(cx - 1, cy + 3)
            check_path.lineTo(cx + 4, cy - 3)
            painter.drawPath(check_path)
        elif self.underMouse():
            ring_pen = QPen(QColor(255, 255, 255, 70), 1.5)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius + 1.5, radius + 1.5)


# ─────────────────────────────────────────────
#  Glass panel container
# ─────────────────────────────────────────────
class GlassPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setBrush(QBrush(Theme.BG_GLASS))
        painter.setPen(QPen(Theme.BG_GLASS_BORDER, 1))
        painter.drawRoundedRect(rect, Theme.RADIUS_LG, Theme.RADIUS_LG)


# ─────────────────────────────────────────────
#  File playback transport
# ─────────────────────────────────────────────
class PlayPauseButton(QPushButton):
    """
    Play/pause button that paints its own glyph.

    The obvious characters for this (U+25B6 / U+23F8) are missing from plenty
    of UI fonts and fall back to a replacement box, so the triangle and the two
    bars are drawn instead of typed.
    """

    SIZE = (34, 28)
    GLYPH_HEIGHT = 12
    BAR_WIDTH = 3.0
    BAR_GAP = 3.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(*self.SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # None until the first set_playing, so that call is never mistaken for
        # a redundant one and always writes the tooltip and accessible name.
        self._is_playing = None
        self.set_playing(False)
        self.setStyleSheet(TOOL_BUTTON_STYLE % {"radius": Theme.RADIUS_SM,
                                                "font_size": Theme.FONT_SIZE_SMALL})

    def set_playing(self, is_playing):
        """Show a pause glyph while playing, a play glyph while stopped."""
        is_playing = bool(is_playing)
        # The render loop reasserts the transport state ~60 times a second, so
        # only a real change is allowed to schedule a repaint.
        if is_playing == self._is_playing:
            return
        self._is_playing = is_playing
        action = "Pause" if self._is_playing else "Play"
        self.setToolTip(f"{action} (Space)")
        self.setAccessibleName(action)
        self.update()

    def paintEvent(self, event):
        # Let the stylesheet paint the pill, then draw the glyph over it.
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(Theme.TEXT_PRIMARY))

        cx, cy = self.width() / 2.0, self.height() / 2.0
        half = self.GLYPH_HEIGHT / 2.0

        if self._is_playing:
            top, height = cy - half, self.GLYPH_HEIGHT
            left = cx - self.BAR_GAP / 2.0 - self.BAR_WIDTH
            painter.drawRoundedRect(QRectF(left, top, self.BAR_WIDTH, height), 1, 1)
            painter.drawRoundedRect(
                QRectF(cx + self.BAR_GAP / 2.0, top, self.BAR_WIDTH, height), 1, 1)
        else:
            # Equilateral-ish triangle, nudged right so it reads as centred.
            width = self.GLYPH_HEIGHT * 0.9
            left = cx - width / 2.0 + 1.0
            triangle = QPainterPath()
            triangle.moveTo(left, cy - half)
            triangle.lineTo(left + width, cy)
            triangle.lineTo(left, cy + half)
            triangle.closeSubpath()
            painter.drawPath(triangle)


class PlaybackBar(GlassPanel):
    """
    Transport for a locally opened audio file: play/pause, a seek slider, the
    elapsed/total time and a button to hand the visualizer back to system audio.

    Hidden until a file is loaded. Like MainWindow, it reports user intent
    through plain callback attributes rather than driving playback itself.
    """

    HEIGHT = 46
    TRACK_LABEL_WIDTH = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setVisible(False)

        # Assigned by the owner to observe transport interactions.
        self.on_play_pause = None
        self.on_seek = None
        self.on_close = None

        # Set while the seek slider is being written from playback position, so
        # that echo is not mistaken for the user scrubbing.
        self._syncing = False
        self._scrubbing = False
        self._duration = 0.0
        self._track_name = ""
        self._shown_position_text = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(10)

        self.play_button = PlayPauseButton()
        self.play_button.clicked.connect(self._emit_play_pause)
        layout.addWidget(self.play_button)

        self.track_label = QLabel("")
        self.track_label.setFont(Theme.font(Theme.FONT_SIZE_SMALL, QFont.Weight.Medium))
        self.track_label.setFixedWidth(self.TRACK_LABEL_WIDTH)
        self.track_label.setStyleSheet(
            "color: rgba(240,240,245,0.9); background: transparent;")
        layout.addWidget(self.track_label)

        self.position_label = self._time_label("0:00")
        layout.addWidget(self.position_label)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setStyleSheet(SLIDER_STYLE)
        self.seek_slider.setToolTip("Seek within the file")
        self.seek_slider.setAccessibleName("Playback position")
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        self.seek_slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.seek_slider, stretch=1)

        self.duration_label = self._time_label("0:00")
        layout.addWidget(self.duration_label)

        self.close_button = tool_button("✕", "Close the file and return to "
                                        "system audio", "Close file",
                                        size=(28, 28))
        self.close_button.clicked.connect(self._emit_close)
        layout.addWidget(self.close_button)

    @staticmethod
    def _time_label(text):
        label = QLabel(text)
        label.setFont(Theme.font(Theme.FONT_SIZE_SMALL))
        label.setStyleSheet("color: rgba(160,160,175,0.9); background: transparent;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumWidth(38)
        return label

    # ── State from the application ──

    def set_track(self, name, duration):
        """Show the bar for `name`, a file `duration` seconds long."""
        self._track_name = name
        self._duration = max(0.0, duration)
        self._elide_track_name()
        self.track_label.setToolTip(name)
        self.duration_label.setText(format_duration(self._duration))

        self._syncing = True
        # Milliseconds keep the slider smooth on short files without needing a
        # separate scale factor for long ones.
        self.seek_slider.setRange(0, max(1, int(self._duration * 1000)))
        self.seek_slider.setValue(0)
        self._syncing = False

        self._shown_position_text = None
        self.set_position(0.0)
        self.setVisible(True)

    def clear_track(self):
        """Hide the bar; no file is loaded."""
        self.setVisible(False)
        self._scrubbing = False
        self._duration = 0.0
        self._track_name = ""
        self.track_label.setText("")
        self.track_label.setToolTip("")

    def set_position(self, seconds):
        """Move the slider to the current playback position."""
        if self._scrubbing:
            return
        self._syncing = True
        self.seek_slider.setValue(int(max(0.0, seconds) * 1000))
        self._syncing = False
        self._show_position(seconds)

    def set_playing(self, is_playing):
        """Swap the button between play and pause affordances."""
        self.play_button.set_playing(is_playing)

    # ── Internals ──

    def _elide_track_name(self):
        metrics = QFontMetrics(self.track_label.font())
        self.track_label.setText(metrics.elidedText(
            self._track_name, Qt.TextElideMode.ElideMiddle,
            self.track_label.width() or self.TRACK_LABEL_WIDTH))

    def _show_position(self, seconds):
        # The timer ticks at ~60 FPS but the text only changes once a second;
        # skipping the no-op writes keeps a relayout off the render path.
        text = format_duration(seconds)
        if text != self._shown_position_text:
            self._shown_position_text = text
            self.position_label.setText(text)

    def _emit_play_pause(self):
        if self.on_play_pause is not None:
            self.on_play_pause()

    def _emit_close(self):
        if self.on_close is not None:
            self.on_close()

    def _emit_seek(self, milliseconds):
        if self.on_seek is not None:
            self.on_seek(milliseconds / 1000.0)

    def _on_slider_pressed(self):
        self._scrubbing = True

    def _on_slider_moved(self, value):
        # Preview the target time while dragging; the seek lands on release.
        self._show_position(value / 1000.0)

    def _on_slider_released(self):
        self._scrubbing = False
        self._emit_seek(self.seek_slider.value())

    def _on_slider_value_changed(self, value):
        # Keyboard and page-step changes never go through press/release, so
        # they are seeked here; drags and app-driven updates are excluded.
        if self._syncing or self._scrubbing:
            return
        self._show_position(value / 1000.0)
        self._emit_seek(value)


# ─────────────────────────────────────────────
#  Custom title bar
# ─────────────────────────────────────────────
class TitleBar(QWidget):
    def __init__(self, parent_window, parent=None):
        super().__init__(parent)
        self.parent_window = parent_window
        self._drag_pos = None
        self.setFixedHeight(42)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 8, 0)
        layout.setSpacing(0)

        # Title label
        title_label = QLabel("Audio Visualizer")
        title_label.setFont(Theme.font(Theme.FONT_SIZE_TITLE, QFont.Weight.DemiBold))
        title_label.setStyleSheet("color: rgba(240,240,245,0.9); background: transparent;")
        layout.addWidget(title_label)

        layout.addStretch()

        # Opening a file lives here rather than in the controls panel: that row
        # is already tight at the window's minimum width, and this one is not.
        self.open_file_button = tool_button(
            "Open &File", "Open a local audio file to play and visualize "
            "(Ctrl+O)", "Open file", size=(78, 26))
        self.open_file_button.clicked.connect(self.parent_window.open_audio_file)
        layout.addWidget(self.open_file_button)

        layout.addSpacing(10)

        # Window buttons
        btn_style_base = """
            QPushButton {{
                border: none;
                border-radius: {radius}px;
                background: rgba(255,255,255,0.06);
                color: rgba(240,240,245,0.7);
                font-size: {font_size}px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background: rgba(255,255,255,0.14);
                color: rgba(240,240,245,0.95);
            }}
        """

        self.min_btn = QPushButton("─")
        self.min_btn.setFixedSize(34, 26)
        self.min_btn.setStyleSheet(btn_style_base.format(radius=6, font_size=12))
        self.min_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.min_btn.setToolTip("Minimize")
        self.min_btn.setAccessibleName("Minimize")
        self.min_btn.clicked.connect(self.parent_window.showMinimized)
        layout.addWidget(self.min_btn)

        layout.addSpacing(4)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(34, 26)
        close_style = """
            QPushButton {
                border: none;
                border-radius: 6px;
                background: rgba(255,255,255,0.06);
                color: rgba(240,240,245,0.7);
                font-size: 11px;
                padding: 0px;
            }
            QPushButton:hover {
                background: rgba(255,60,60,0.75);
                color: rgba(255,255,255,0.95);
            }
        """
        self.close_btn.setStyleSheet(close_style)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setToolTip("Close")
        self.close_btn.setAccessibleName("Close")
        self.close_btn.clicked.connect(self.parent_window.close)
        layout.addWidget(self.close_btn)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.parent_window.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.parent_window.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.parent_window.isMaximized():
            self.parent_window.showNormal()
        else:
            self.parent_window.showMaximized()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Subtle bottom border
        painter.setPen(QPen(QColor(255, 255, 255, 10), 1))
        painter.drawLine(16, self.height() - 1, self.width() - 16, self.height() - 1)


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    # (button label with mnemonic, mode name, stack index)
    MODES = (("&Bars", "Bars", 0),
             ("&Wave", "Waveform", 1),
             ("&Spectrum", "Spectrogram", 2))

    DEFAULT_COLOR_NAME = "Cyan"

    BAR_COUNT_MIN = 8
    BAR_COUNT_MAX = 128
    BAR_COUNT_STEP = 8
    BAR_COUNT_DEFAULT = 32

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer")
        self.resize(900, 640)
        self.setMinimumSize(600, 400)

        # Frameless window
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.bar_widget = BarVisualizerWidget()
        self.waveform_widget = WaveformWidget()
        self.spectrogram_widget = SpectrogramWidget()

        self.current_mode = "Bars"
        self._current_color_name = self.DEFAULT_COLOR_NAME

        # Assigned by the application to observe control changes.
        self.on_bars_changed_callback = None
        self.on_mode_changed_callback = None
        self.on_file_opened_callback = None
        self.on_play_pause_callback = None
        self.on_seek_callback = None
        self.on_close_file_callback = None

        # Directory the file dialog reopens in, so picking a second track from
        # the same album is not a fresh walk from the home directory.
        self._last_open_dir = ""

        self._build_ui()
        self._build_shortcuts()

    def _build_ui(self):
        # Root widget with rounded dark background
        root = QWidget()
        root.setObjectName("rootWidget")
        root.setStyleSheet("""
            #rootWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0d0d16, stop:1 #08080e);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.05);
            }
        """)
        self.setCentralWidget(root)

        outer_layout = QVBoxLayout(root)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Title bar
        self.title_bar = TitleBar(self)
        outer_layout.addWidget(self.title_bar)

        # Main content area
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 4, 16, 14)
        content_layout.setSpacing(12)

        # Visualization stack
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background: transparent; border-radius: 10px;")
        self.stack.addWidget(self.bar_widget)         # index 0
        self.stack.addWidget(self.waveform_widget)     # index 1
        self.stack.addWidget(self.spectrogram_widget)  # index 2
        content_layout.addWidget(self.stack, stretch=1)

        # ── File playback transport (hidden until a file is opened) ──
        self.playback_bar = PlaybackBar()
        self.playback_bar.on_play_pause = self._on_play_pause_clicked
        self.playback_bar.on_seek = self._on_seek_requested
        self.playback_bar.on_close = self._on_close_file_clicked
        content_layout.addWidget(self.playback_bar)

        # ── Controls panel (glass) ──
        glass = GlassPanel()
        glass.setFixedHeight(72)
        glass_layout = QHBoxLayout(glass)
        glass_layout.setContentsMargins(16, 10, 16, 10)
        glass_layout.setSpacing(20)

        glass_layout.addWidget(self._build_mode_control())
        glass_layout.addWidget(self._separator())
        glass_layout.addWidget(self._build_bar_count_control())
        glass_layout.addWidget(self._separator())
        glass_layout.addLayout(self._build_color_control())
        glass_layout.addStretch()
        glass_layout.addWidget(self._build_status_label())

        content_layout.addWidget(glass)
        outer_layout.addWidget(content, stretch=1)

        # Reflect the default selection in the visualizers. Without this the
        # active swatch and the rendered color would disagree until first click.
        self._apply_current_color()

    # ── Control construction ──

    def _separator(self):
        sep = QWidget()
        sep.setFixedSize(1, 32)
        sep.setStyleSheet("background: rgba(255,255,255,0.1);")
        return sep

    def _control_label(self, text):
        label = QLabel(text)
        label.setFont(Theme.font(Theme.FONT_SIZE_SMALL))
        label.setStyleSheet("color: rgba(160,160,175,0.9); background: transparent;")
        return label

    def _build_mode_control(self):
        mode_container = QWidget()
        mode_container.setStyleSheet(f"""
            background: rgba(255,255,255,0.05);
            border-radius: {Theme.RADIUS_MD}px;
            border: 1px solid rgba(255,255,255,0.08);
        """)
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(3, 3, 3, 3)
        mode_layout.setSpacing(2)

        self.mode_buttons = []
        for label, mode, _index in self.MODES:
            btn = SegmentButton(label)
            # The mnemonic in `label` gives each mode an Alt+key shortcut; the
            # accessible name keeps screen readers off the raw marker text.
            btn.setAccessibleName(mode)
            btn.setToolTip(f"Show the {mode} visualization "
                           f"(Alt+{label[label.index('&') + 1]})")
            btn.mode = mode
            btn.clicked.connect(lambda checked, m=mode: self._on_segment_clicked(m))
            mode_layout.addWidget(btn)
            self.mode_buttons.append(btn)

        self._sync_mode_buttons()
        return mode_container

    def _build_bar_count_control(self):
        self.bar_count_container = QWidget()
        bar_count_layout = QHBoxLayout(self.bar_count_container)
        bar_count_layout.setContentsMargins(0, 0, 0, 0)
        bar_count_layout.setSpacing(8)

        bars_label = self._control_label("&Count")
        bar_count_layout.addWidget(bars_label)

        self.bar_slider = QSlider(Qt.Orientation.Horizontal)
        self.bar_slider.setRange(self.BAR_COUNT_MIN, self.BAR_COUNT_MAX)
        self.bar_slider.setValue(self.BAR_COUNT_DEFAULT)
        self.bar_slider.setSingleStep(self.BAR_COUNT_STEP)
        self.bar_slider.setPageStep(self.BAR_COUNT_STEP * 2)
        self.bar_slider.setFixedWidth(100)
        self.bar_slider.setToolTip("Adjust the number of frequency bars (Alt+C)")
        self.bar_slider.setAccessibleName("Bar count")
        bars_label.setBuddy(self.bar_slider)
        self.bar_slider.setStyleSheet(SLIDER_STYLE)
        self.bar_slider.valueChanged.connect(self._on_bars_changed)
        bar_count_layout.addWidget(self.bar_slider)

        self.bar_count_label = QLabel(str(self.BAR_COUNT_DEFAULT))
        self.bar_count_label.setFont(Theme.font(Theme.FONT_SIZE_SMALL, QFont.Weight.Medium))
        self.bar_count_label.setFixedWidth(28)
        self.bar_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bar_count_label.setStyleSheet("color: rgba(240,240,245,0.85); background: transparent;")
        bar_count_layout.addWidget(self.bar_count_label)

        return self.bar_count_container

    def _build_color_control(self):
        color_layout = QHBoxLayout()
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(8)

        color_label = self._control_label("C&olor")
        color_layout.addWidget(color_label)

        self.color_map = {
            "Rainbow": None,
            "Cyan":    QColor(0, 200, 255),
            "Red":     QColor(255, 65, 65),
            "Orange":  QColor(255, 145, 50),
            "Yellow":  QColor(255, 220, 55),
            "Green":   QColor(65, 230, 100),
            "Blue":    QColor(55, 90, 255),
            "Purple":  QColor(160, 80, 255),
            "Magenta": QColor(255, 55, 200),
        }

        swatch_layout = QHBoxLayout()
        swatch_layout.setContentsMargins(0, 0, 0, 0)
        swatch_layout.setSpacing(4)

        self.swatches = []
        for name, color in self.color_map.items():
            swatch = ColorSwatch(color, name, is_rainbow=(color is None))
            swatch.clicked.connect(lambda checked, n=name: self._on_color_swatch_clicked(n))
            swatch_layout.addWidget(swatch)
            self.swatches.append(swatch)

        # Alt+O reaches the swatch strip; arrow keys move within it.
        color_label.setBuddy(self.swatches[0])
        self._sync_swatches()

        color_layout.addLayout(swatch_layout)
        return color_layout

    def _build_shortcuts(self):
        open_shortcut = QShortcut(QKeySequence.StandardKey.Open, self)
        open_shortcut.activated.connect(self.open_audio_file)

        # Space is the conventional play/pause key, but a window-scoped
        # shortcut takes it from whichever button has focus. It is therefore
        # only live while a file is open; the rest of the time Space keeps
        # activating the focused control as usual.
        self.play_pause_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.play_pause_shortcut.activated.connect(self._on_play_pause_clicked)
        self.play_pause_shortcut.setEnabled(False)

    # ── File playback ──

    def open_audio_file(self):
        """Ask for an audio file and hand the choice to the application."""
        path, _selected_filter = QFileDialog.getOpenFileName(
            self, "Open Audio File", self._last_open_dir, file_dialog_filter())
        if not path:
            return
        self._last_open_dir = os.path.dirname(path)
        if self.on_file_opened_callback is not None:
            self.on_file_opened_callback(path)

    def set_playback_file(self, name, duration):
        """Show the transport for a newly loaded file."""
        self.playback_bar.set_track(name, duration)
        self.playback_bar.set_playing(True)
        self.play_pause_shortcut.setEnabled(True)

    def clear_playback_file(self):
        """Hide the transport; the visualizer is back on system audio."""
        self.playback_bar.clear_track()
        self.play_pause_shortcut.setEnabled(False)

    def set_playback_position(self, seconds):
        self.playback_bar.set_position(seconds)

    def set_playback_state(self, is_playing):
        self.playback_bar.set_playing(is_playing)

    def _on_play_pause_clicked(self):
        if self.on_play_pause_callback is not None:
            self.on_play_pause_callback()

    def _on_seek_requested(self, seconds):
        if self.on_seek_callback is not None:
            self.on_seek_callback(seconds)

    def _on_close_file_clicked(self):
        if self.on_close_file_callback is not None:
            self.on_close_file_callback()

    def _build_status_label(self):
        self.status_label = QLabel("")
        self.status_label.setFont(Theme.font(Theme.FONT_SIZE_SMALL))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: rgba(160,160,175,0.9); background: transparent;")
        self.status_label.setVisible(False)
        return self.status_label

    def set_status(self, message, is_error=False):
        """Show a short status message in the controls panel, or clear it."""
        self.status_label.setText(message or "")
        self.status_label.setVisible(bool(message))
        color = Theme.TEXT_ERROR if is_error else Theme.TEXT_SECONDARY
        self.status_label.setStyleSheet(
            f"color: rgba({color.red()},{color.green()},{color.blue()},0.95); "
            "background: transparent;")
        if message:
            self.status_label.setToolTip(message)

    # ── Helpers ──

    def _active_widget(self):
        return self.stack.currentWidget()

    def _sync_mode_buttons(self):
        for btn in self.mode_buttons:
            btn.setChecked(btn.mode == self.current_mode)

    def _sync_swatches(self):
        for swatch in self.swatches:
            swatch.set_active(swatch.swatch_name == self._current_color_name)

    def _on_segment_clicked(self, mode):
        index = next((i for _label, m, i in self.MODES if m == mode), 0)

        self.current_mode = mode
        self._sync_mode_buttons()
        self.stack.setCurrentIndex(index)

        # Show bar count only in Bars mode
        self.bar_count_container.setVisible(mode == "Bars")

        if self.on_mode_changed_callback is not None:
            self.on_mode_changed_callback(self.current_mode)

        # Re-apply color to new widget
        self._apply_current_color()

    def _on_bars_changed(self, value):
        # Snap to multiples of the step size
        snapped = max(self.BAR_COUNT_MIN,
                      (value // self.BAR_COUNT_STEP) * self.BAR_COUNT_STEP)
        if snapped != value:
            # Write the snapped value back so the slider and the rendered bar
            # count never disagree; blocked to avoid re-entering this slot.
            self.bar_slider.blockSignals(True)
            self.bar_slider.setValue(snapped)
            self.bar_slider.blockSignals(False)
        self.bar_count_label.setText(str(snapped))
        self.bar_widget.set_bars(snapped)
        if self.on_bars_changed_callback is not None:
            self.on_bars_changed_callback(snapped)

    def _on_color_swatch_clicked(self, color_name):
        self._current_color_name = color_name
        self._sync_swatches()
        self._apply_current_color()

    def _apply_current_color(self):
        color = self.color_map.get(self._current_color_name)
        for widget in [self.bar_widget, self.waveform_widget, self.spectrogram_widget]:
            if color is None:
                widget.set_rainbow_mode(True)
            else:
                widget.set_color(color)

    def paintEvent(self, event):
        # Needed for transparent frameless window
        pass
