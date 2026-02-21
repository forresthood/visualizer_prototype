"""
Unit tests for spectrogram generation functionality in AudioProcessor.
"""
import unittest
import numpy as np
from audio_processing import AudioProcessor

class TestSpectrogramGeneration(unittest.TestCase):
    """Test suite for spectrogram data generation (get_raw_fft)."""

    def setUp(self):
        self.sample_rate = 44100
        self.buffer_frames = 1024
        self.processor = AudioProcessor(sample_rate=self.sample_rate, buffer_frames=self.buffer_frames)

    def test_input_none(self):
        """Test that None input returns an array of zeros."""
        fft_data = self.processor.get_raw_fft(None)
        expected_len = self.buffer_frames // 2 + 1
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data == 0))

    def test_input_empty(self):
        """Test that empty input returns an array of zeros."""
        fft_data = self.processor.get_raw_fft([])
        expected_len = self.buffer_frames // 2 + 1
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data == 0))

    def test_input_short_padding(self):
        """Test that input shorter than buffer_frames is handled correctly (padded)."""
        # Create a short signal: 100 frames of silence
        # We want to make sure it doesn't crash
        short_signal = np.zeros(100)
        fft_data = self.processor.get_raw_fft(short_signal)
        expected_len = self.buffer_frames // 2 + 1
        self.assertEqual(len(fft_data), expected_len)
        # Should be all zeros (or close to it)
        # Since it's silence, it should be clipped to 0
        self.assertTrue(np.all(fft_data == 0))

    def test_input_short_sine(self):
        """Test that a short sine wave produces output."""
        # Create a short sine wave: 512 frames (half buffer)
        t = np.linspace(0, 512 / self.sample_rate, 512, endpoint=False)
        freq = 1000
        short_signal = 0.5 * np.sin(2 * np.pi * freq * t)

        fft_data = self.processor.get_raw_fft(short_signal)
        # Should have some energy, not all zeros
        self.assertFalse(np.all(fft_data == 0))
        self.assertTrue(np.max(fft_data) > 0)

    def test_input_long_truncation(self):
        """Test that input longer than buffer_frames is handled correctly (truncated)."""
        # Create a long signal: 2000 frames
        long_signal = np.zeros(2000)
        fft_data = self.processor.get_raw_fft(long_signal)
        expected_len = self.buffer_frames // 2 + 1
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data == 0))

    def test_input_long_truncation_logic(self):
        """Test that truncation takes the *last* buffer_frames."""
        # Create a signal where the first part is silence, last part is a sine wave
        # If truncation takes the last part, we should see energy.
        # If it takes the first part, we should see silence.

        # 2000 frames total. First 1000 silence. Last 1000 sine wave.
        # buffer_frames = 1024.
        # If we take last 1024, we get 24 frames of silence + 1000 frames of sine.

        total_len = 2000
        sine_len = 1000
        silence_len = total_len - sine_len

        silence = np.zeros(silence_len)
        t = np.linspace(0, sine_len / self.sample_rate, sine_len, endpoint=False)
        freq = 1000
        sine = 0.5 * np.sin(2 * np.pi * freq * t)

        long_signal = np.concatenate((silence, sine))

        fft_data = self.processor.get_raw_fft(long_signal)

        # Should have energy because the end contains the signal
        self.assertFalse(np.all(fft_data == 0))
        self.assertTrue(np.max(fft_data) > 0)

    def test_output_range(self):
        """Test that output values are strictly within [0.0, 1.0]."""
        # Use random noise to generate various frequencies and amplitudes
        np.random.seed(42)
        noise = np.random.uniform(-1.0, 1.0, self.buffer_frames)
        fft_data = self.processor.get_raw_fft(noise)

        self.assertTrue(np.all(fft_data >= 0.0))
        self.assertTrue(np.all(fft_data <= 1.0))

    def test_silence(self):
        """Test that silence input results in minimal values."""
        silence = np.zeros(self.buffer_frames)
        fft_data = self.processor.get_raw_fft(silence)
        # Should be all zeros due to clipping of low dB values
        self.assertTrue(np.all(fft_data == 0.0))

    def test_amplitude_scaling(self):
        """Test that higher amplitude input results in higher magnitude output."""
        # Create two sine waves with different amplitudes
        t = np.linspace(0, self.buffer_frames / self.sample_rate, self.buffer_frames, endpoint=False)
        freq = 1000

        signal_low = 0.1 * np.sin(2 * np.pi * freq * t)
        signal_high = 0.8 * np.sin(2 * np.pi * freq * t)

        fft_low = self.processor.get_raw_fft(signal_low)
        fft_high = self.processor.get_raw_fft(signal_high)

        # Find peak bin
        peak_bin = np.argmax(fft_high)

        # High amplitude signal should have higher value at peak
        self.assertGreater(fft_high[peak_bin], fft_low[peak_bin])

    def test_dc_offset(self):
        """Test that a DC offset (constant value) results in energy at bin 0."""
        # Constant signal
        dc_signal = np.full(self.buffer_frames, 0.5)
        fft_data = self.processor.get_raw_fft(dc_signal)

        # Max energy should be at DC (bin 0) or close to it
        peak_bin = np.argmax(fft_data)
        self.assertEqual(peak_bin, 0)

if __name__ == '__main__':
    unittest.main()
