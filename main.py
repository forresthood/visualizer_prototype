import sys
import numpy as np

# Monkey-patch numpy.fromstring for soundcard compatibility with NumPy 2.x
# (In NumPy 2.0, fromstring still exists but raises an error when reading binary data)
np.fromstring = np.frombuffer

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from audio_capture import AudioCaptureSubsystem
from audio_processing import AudioProcessor
from visualizer_ui import MainWindow

class AudioVisualizerApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        
        # Initialize Audio Capture
        self.audio_capture = AudioCaptureSubsystem(sample_rate=44100, buffer_frames=2048)
        
        # Initialize Processor
        self.processor = AudioProcessor(sample_rate=44100, buffer_frames=2048)
        
        # Initialize UI
        self.main_window = MainWindow()
        self.num_bars = self.main_window.bar_widget.bars
        self.current_mode = "Bars"
        
        # Connect UI callbacks
        self.main_window.on_bars_changed_callback = self.set_num_bars
        self.main_window.on_mode_changed_callback = self.set_mode

        # Setup Update Timer (~60 FPS -> ~16ms)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualizer)
        self.timer.start(16)

    def set_num_bars(self, count):
        self.num_bars = count
        
    def set_mode(self, mode_name):
        self.current_mode = mode_name
        
    def update_visualizer(self):
        # Get latest audio data
        audio_data = self.audio_capture.get_latest_data()
        
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
        
        sys.exit(exit_code)

if __name__ == "__main__":
    try:
        app = AudioVisualizerApp()
        app.run()
    except Exception as e:
        print(f"Failed to start the application: {e}")
