import os
import sys
import traceback
import numpy as np

# Monkey-patch numpy.fromstring for soundcard compatibility with NumPy 2.x
# (In NumPy 2.0, fromstring still exists but raises an error when reading binary data)
np.fromstring = np.frombuffer

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from audio_capture import AudioCaptureSubsystem
from audio_file import AudioFileError, AudioFilePlayer
from audio_processing import AudioProcessor
from visualizer_ui import MainWindow

SAMPLE_RATE = 44100
BUFFER_FRAMES = 2048

# The two things the visualizer can be looking at.
SOURCE_SYSTEM = "system"
SOURCE_FILE = "file"

class AudioVisualizerApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        
        # Initialize Audio Capture
        self.audio_capture = AudioCaptureSubsystem(sample_rate=SAMPLE_RATE,
                                                   buffer_frames=BUFFER_FRAMES)

        # Playback of a local file, used instead of the capture when the user
        # opens one.
        self.file_player = AudioFilePlayer(chunk_frames=BUFFER_FRAMES)
        self.source = SOURCE_SYSTEM

        # Initialize Processor
        self.processor = AudioProcessor(sample_rate=SAMPLE_RATE,
                                        buffer_frames=BUFFER_FRAMES)
        
        # Initialize UI
        self.main_window = MainWindow()
        self.num_bars = self.main_window.bar_widget.bars
        self.current_mode = self.main_window.current_mode
        self._reported_error = None
        
        # Connect UI callbacks
        self.main_window.on_bars_changed_callback = self.set_num_bars
        self.main_window.on_mode_changed_callback = self.set_mode
        self.main_window.on_file_opened_callback = self.open_file
        self.main_window.on_play_pause_callback = self.toggle_playback
        self.main_window.on_seek_callback = self.seek
        self.main_window.on_close_file_callback = self.close_file

        # Setup Update Timer (~60 FPS -> ~16ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualizer)
        self.timer.start(16)

    def set_num_bars(self, count):
        self.num_bars = count
        
    def set_mode(self, mode_name):
        self.current_mode = mode_name

    # ── Source selection ──

    def open_file(self, path):
        """Decode `path`, play it, and visualize it instead of system audio."""
        try:
            self.file_player.load(path)
        except AudioFileError as e:
            self.main_window.set_status(str(e), is_error=True)
            return

        # Stop the capture: the file is the source now, and a loopback device
        # would otherwise re-capture this very playback on top of it.
        self.audio_capture.stop()
        self.source = SOURCE_FILE

        # Files bring their own sample rate; retune the analysis to it rather
        # than resampling the audio we are about to play.
        self.processor.set_sample_rate(self.file_player.sample_rate)

        self._reported_error = None
        self.main_window.set_status("")
        self.main_window.set_playback_file(os.path.basename(path),
                                           self.file_player.duration)

    def close_file(self):
        """Unload the file and return the visualizer to system audio."""
        self.file_player.unload()
        self.source = SOURCE_SYSTEM
        self.processor.set_sample_rate(SAMPLE_RATE)

        self._reported_error = None
        self.main_window.set_status("")
        self.main_window.clear_playback_file()
        self.audio_capture.start()

    def toggle_playback(self):
        if self.source == SOURCE_FILE:
            self.main_window.set_playback_state(self.file_player.toggle())

    def seek(self, seconds):
        if self.source == SOURCE_FILE:
            self.file_player.seek(seconds)
            self.main_window.set_playback_position(self.file_player.position)

    def _check_capture_health(self):
        """Surface a dead capture thread instead of showing a frozen visualizer."""
        error = self.audio_capture.last_error
        if error != self._reported_error:
            self._reported_error = error
            self.main_window.set_status(
                f"Audio capture stopped — {error}" if error else "",
                is_error=bool(error))

    def _check_playback_health(self):
        """Surface a dead playback thread, e.g. an output device disappearing."""
        error = self.file_player.last_error
        if error != self._reported_error:
            self._reported_error = error
            self.main_window.set_status(
                f"Playback stopped — {error}" if error else "",
                is_error=bool(error))

    def _poll_source(self):
        """Service the active source and return its latest audio chunk."""
        if self.source == SOURCE_FILE:
            self._check_playback_health()
            self.main_window.set_playback_state(self.file_player.is_playing)
            self.main_window.set_playback_position(self.file_player.position)
            return self.file_player.get_latest_data()

        self._check_capture_health()
        return self.audio_capture.get_latest_data()

    def update_visualizer(self):
        audio_data = self._poll_source()

        if audio_data is not None:
            if self.current_mode == "Bars":
                bins = self.processor.compute_fft(audio_data, self.num_bars)
                self.main_window.bar_widget.update_values(bins)
            elif self.current_mode == "Waveform":
                self.main_window.waveform_widget.update_waveform(audio_data)
            elif self.current_mode == "Spectrogram":
                fft_data = self.processor.get_raw_fft(audio_data)
                self.main_window.spectrogram_widget.update_fft(fft_data)
            
    def run(self):
        print("Starting Audio Capture...")
        self.audio_capture.start()
        
        print("Starting UI...")
        self.main_window.show()
        
        exit_code = self.app.exec()
        
        print("Stopping Audio Capture...")
        self.audio_capture.stop()
        self.file_player.unload()
        
        return exit_code

if __name__ == "__main__":
    try:
        app = AudioVisualizerApp()
        sys.exit(app.run())
    except Exception:
        # Print the full traceback and exit non-zero: a startup failure must not
        # look like a successful run to CI or a launcher script.
        print("Failed to start the application:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
