import sys
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox,
                             QSpinBox, QGroupBox, QStackedWidget, QPushButton)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QLinearGradient,
                          QPainterPath, QRadialGradient, QMouseEvent, QFont, QImage)

# ─────────────────────────────────────────────
#  Theme Constants
# ─────────────────────────────────────────────
class Theme:
    VIZ_BG = QColor(13, 13, 22)
    BG_GLASS = QColor(255, 255, 255, 10)
    BG_GLASS_BORDER = QColor(255, 255, 255, 20)
    RADIUS_LG = 12
    RADIUS_SM = 6
    FONT_SIZE_TITLE = 16
    FONT_SIZE_SMALL = 10
    SEG_ACTIVE_BG = QColor(255, 255, 255, 20)
    SEG_BORDER = QColor(255, 255, 255, 40)
    TEXT_PRIMARY = QColor(240, 240, 245)
    TEXT_SECONDARY = QColor(180, 180, 190)

    @staticmethod
    def font(size, weight=QFont.Weight.Normal):
        f = QFont("Segoe UI", size)
        f.setWeight(weight)
        return f

# ─────────────────────────────────────────────
#  Base mixin for shared color / rainbow logic
# ─────────────────────────────────────────────
class ColorMixin:
    """Shared color and rainbow state for all visualizer widgets."""
    def init_color(self):
        self.bar_color = QColor(88, 130, 255)
        self.bg_color = Theme.VIZ_BG
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

        for i in range(self.bars):
            val = max(0.0, min(1.0, self.bar_values[i]))
            bar_height = val * draw_h
            x = spacing + i * (bar_width + spacing)
            y = h - padding_bottom - bar_height
            rect = QRectF(x, y, bar_width, bar_height)

            # Determine color
            if self.rainbow_mode:
                current_color = QColor.fromHsvF((self.rainbow_hue + i / self.bars) % 1.0, 0.75, 1.0)
            else:
                current_color = self.bar_color

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
            gradient.setColorAt(0, current_color.darker(160))
            gradient.setColorAt(0.5, current_color)
            gradient.setColorAt(1, current_color.lighter(130))
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

        # Subtle grid lines
        grid_pen = QPen(QColor(255, 255, 255, 12), 1)
        painter.setPen(grid_pen)
        for frac in [0.25, 0.5, 0.75]:
            gy = int(h * frac)
            painter.drawLine(0, gy, w, gy)

        # Build waveform path
        path = QPainterPath()
        step = w / n
        for i in range(n):
            x = i * step
            y = mid_y - self.waveform_data[i] * mid_y * 0.9
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        if self.rainbow_mode:
            segment_count = min(n, 200)
            segment_size = max(1, n // segment_count)

            # Glow pass for rainbow
            for seg in range(0, n, segment_size):
                glow_path = QPainterPath()
                hue = (self.rainbow_hue + seg / n) % 1.0
                glow_color = QColor.fromHsvF(hue, 0.7, 1.0, 0.25)
                for j in range(seg, min(seg + segment_size + 1, n)):
                    x = j * step
                    y = mid_y - self.waveform_data[j] * mid_y * 0.9
                    if j == seg:
                        glow_path.moveTo(x, y)
                    else:
                        glow_path.lineTo(x, y)
                painter.setPen(QPen(glow_color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawPath(glow_path)

            # Crisp pass for rainbow
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
            # Glow pass
            glow_color = QColor(self.bar_color)
            glow_color.setAlpha(50)
            painter.setPen(QPen(glow_color, 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(path)

            # Crisp pass
            painter.setPen(QPen(self.bar_color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(path)


# ─────────────────────────────────────────────
#  3. Spectrogram Visualizer
# ─────────────────────────────────────────────
class SpectrogramWidget(ColorMixin, QWidget):
    def __init__(self, parent=None, history_length=200):
        super().__init__(parent)
        self.init_color()
        self.history_length = history_length
        self.fft_history = deque(maxlen=history_length)
        self._viridis_lut = None

    def update_fft(self, fft_data):
        self.fft_history.append(fft_data)
        self.update()

    def _build_viridis_lut(self):
        lut = []
        for i in range(256):
            t = i / 255.0
            if t < 0.01:
                lut.append(0) # Transparent
            else:
                c = self._viridis_color(t)
                lut.append(c.rgba())
        return lut

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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.bg_color)

        if not self.fft_history:
            return

        try:
            # Convert history to numpy array (history, bins)
            data = np.array(self.fft_history)
            if data.ndim != 2:
                return
        except Exception:
            return

        h_len, n_bins = data.shape
        if h_len == 0 or n_bins == 0:
            return

        if self.rainbow_mode:
            self._paint_rainbow(painter, data, h_len, n_bins)
        else:
            self._paint_viridis(painter, data, h_len, n_bins)

    def _paint_viridis(self, painter, data, w, h):
        if self._viridis_lut is None:
            self._viridis_lut = self._build_viridis_lut()

        # Transpose to (bins, history) so row=bin
        # Flip vertically so bin 0 is at bottom
        img_data = np.flipud(data.T)

        # Scale to 0-255
        img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)

        if not img_data.flags['C_CONTIGUOUS']:
            img_data = np.ascontiguousarray(img_data)

        image = QImage(img_data.data, w, h, w, QImage.Format.Format_Indexed8)
        image.setColorTable(self._viridis_lut)

        painter.drawImage(self.rect(), image)

    def _paint_rainbow(self, painter, data, w, h):
        img_data_source = np.flipud(data.T)
        buffer = np.zeros((h, w, 4), dtype=np.uint8)

        for r_idx in range(h):
            bin_idx = h - 1 - r_idx
            hue = (self.rainbow_hue + bin_idx / h) % 1.0

            c = QColor.fromHsvF(hue, 0.8, 1.0)
            red, green, blue = c.red(), c.green(), c.blue()

            row = img_data_source[r_idx]
            mask = row >= 0.01
            vals = row[mask]

            # BB GG RR AA (Little Endian for 0xAARRGGBB)
            buffer[r_idx, mask, 0] = (blue * vals).astype(np.uint8)
            buffer[r_idx, mask, 1] = (green * vals).astype(np.uint8)
            buffer[r_idx, mask, 2] = (red * vals).astype(np.uint8)
            buffer[r_idx, mask, 3] = 255

        image = QImage(buffer.data, w, h, w * 4, QImage.Format.Format_ARGB32)
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
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())


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
            # Draw a rainbow gradient circle
            for angle in range(360):
                hue = angle / 360.0
                color = QColor.fromHsvF(hue, 0.85, 1.0)
                painter.setPen(QPen(color, 2))
                import math
                px = cx + (radius - 1) * math.cos(math.radians(angle))
                py = cy + (radius - 1) * math.sin(math.radians(angle))
                painter.drawPoint(int(px), int(py))
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
        rect = self.rect().adjusted(0, 0, 0, 0)
        painter.setBrush(QBrush(Theme.BG_GLASS))
        painter.setPen(QPen(Theme.BG_GLASS_BORDER, 1))
        painter.drawRoundedRect(rect, Theme.RADIUS_LG, Theme.RADIUS_LG)


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
        title_label.setStyleSheet(f"color: rgba(240,240,245,0.9); background: transparent;")
        layout.addWidget(title_label)

        layout.addStretch()

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

        # Compat reference
        self.visualizer = self.bar_widget
        self.current_mode = "Bars"

        self._build_ui()

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

        # Default to Cyan (index 1)
        self.swatches[1].set_active(True)

        glass_layout.addLayout(swatch_layout)
        glass_layout.addStretch()

        content_layout.addWidget(glass)
        outer_layout.addWidget(content, stretch=1)

    # ── Helpers ──

    def _active_widget(self):
        return self.stack.currentWidget()

    def _on_segment_clicked(self, label):
        mode_map = {"Bars": "Bars", "Wave": "Waveform", "Spectrum": "Spectrogram"}
        index_map = {"Bars": 0, "Wave": 1, "Spectrum": 2}

        for btn in self.mode_buttons:
            btn.setChecked(btn.text() == label)

        self.current_mode = mode_map.get(label, "Bars")
        self.stack.setCurrentIndex(index_map.get(label, 0))

        # Show bar count only in Bars mode
        self.bar_count_container.setVisible(label == "Bars")

        if hasattr(self, "on_mode_changed_callback"):
            self.on_mode_changed_callback(self.current_mode)

        # Re-apply color to new widget
        self._apply_current_color()

    def _on_bars_changed(self, value):
        # Snap to multiples of 8
        snapped = max(8, (value // 8) * 8)
        if snapped != value:
            self.bar_slider.blockSignals(True)
            self.bar_slider.setValue(snapped)
            self.bar_slider.blockSignals(False)
        self.bar_count_label.setText(str(snapped))
        self.bar_widget.set_bars(snapped)
        if hasattr(self, "on_bars_changed_callback"):
            self.on_bars_changed_callback(snapped)

    def _on_color_swatch_clicked(self, color_name):
        for swatch in self.swatches:
            swatch.set_active(swatch.swatch_name == color_name)
        self._current_color_name = color_name
        self._apply_current_color()

    def _apply_current_color(self):
        name = getattr(self, '_current_color_name', 'Cyan')
        color = self.color_map.get(name)
        for widget in [self.bar_widget, self.waveform_widget, self.spectrogram_widget]:
            if color is None:
                widget.set_rainbow_mode(True)
            else:
                widget.set_color(color)

    def paintEvent(self, event):
        # Needed for transparent frameless window
        pass
