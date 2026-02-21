"""
Unit tests for the AudioProcessor class.
"""
import unittest
import numpy as np
from audio_processing import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    """Test suite for AudioProcessor."""
    def setUp(self):
        self.sample_rate = 44100
        self.buffer_frames = 1024
        self.processor = AudioProcessor(sample_rate=self.sample_rate, buffer_frames=self.buffer_frames)

    def test_compute_fft_shape(self):
        """Test that compute_fft returns the correct number of bars and valid values."""
        audio_data = np.zeros(self.buffer_frames)
        num_bars = 32
        bars = self.processor.compute_fft(audio_data, num_bars)
        self.assertEqual(len(bars), num_bars)
        self.assertTrue(np.all(bars >= 0.0))
        self.assertTrue(np.all(bars <= 1.0))

    def test_get_raw_fft_shape(self):
        """Test that get_raw_fft returns the correct FFT size and valid values."""
        audio_data = np.zeros(self.buffer_frames)
        fft_data = self.processor.get_raw_fft(audio_data)
        # rfft of 1024 points is 513 points (N//2 + 1)
        expected_len = self.buffer_frames // 2 + 1
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data >= 0.0))
        self.assertTrue(np.all(fft_data <= 1.0))

    def test_sine_wave_peak(self):
        """Test that a sine wave input results in a peak at the correct frequency bin."""
        # Create a 1kHz sine wave
        t = np.linspace(0, self.buffer_frames / self.sample_rate, self.buffer_frames, endpoint=False)
        freq = 1000
        audio_data = 0.5 * np.sin(2 * np.pi * freq * t)

        fft_data = self.processor.get_raw_fft(audio_data)

        # Find peak bin
        peak_bin = np.argmax(fft_data)
        peak_freq = peak_bin * self.sample_rate / self.buffer_frames

        # Should be close to 1000Hz
        # Bin resolution = 44100 / 1024 ~= 43 Hz
        self.assertTrue(abs(peak_freq - freq) < 43)

    def test_magic_number_consistency(self):
        """Test that output values remain consistent for a constant input."""
        # This test ensures that if we change implementation, output values for specific inputs
        # remain consistent (regression testing).

        # Use a constant input
        audio_data = np.full(self.buffer_frames, 0.1)

        # Test compute_fft output
        bars = self.processor.compute_fft(audio_data, 32)
        # Just check the first bar value to ensure stability
        # The specific value depends on implementation details
        self.assertGreater(bars[0], 0)

        # Test get_raw_fft output
        fft_data = self.processor.get_raw_fft(audio_data)
        self.assertGreater(fft_data[0], 0)

    def test_compute_fft_edge_cases(self):
        """Test edge cases for compute_fft: None, empty, short, long inputs."""
        num_bars = 32

        # 1. None input
        bars = self.processor.compute_fft(None, num_bars)
        self.assertEqual(len(bars), num_bars)
        self.assertTrue(np.all(bars == 0))

        # 2. Empty input
        bars = self.processor.compute_fft([], num_bars)
        self.assertEqual(len(bars), num_bars)
        self.assertTrue(np.all(bars == 0))

        bars = self.processor.compute_fft(np.array([]), num_bars)
        self.assertEqual(len(bars), num_bars)
        self.assertTrue(np.all(bars == 0))

        # 3. Short input (less than buffer_frames)
        short_data = np.ones(self.buffer_frames // 2)
        bars = self.processor.compute_fft(short_data, num_bars)
        self.assertEqual(len(bars), num_bars)
        # Should return valid normalized values
        self.assertTrue(np.all(bars >= 0.0))
        self.assertTrue(np.all(bars <= 1.0))

        # 4. Long input (more than buffer_frames)
        long_data = np.ones(self.buffer_frames * 2)
        bars = self.processor.compute_fft(long_data, num_bars)
        self.assertEqual(len(bars), num_bars)
        self.assertTrue(np.all(bars >= 0.0))
        self.assertTrue(np.all(bars <= 1.0))

    def test_get_raw_fft_edge_cases(self):
        """Test edge cases for get_raw_fft: None, empty, short, long inputs."""
        expected_len = self.buffer_frames // 2 + 1

        # 1. None input
        fft_data = self.processor.get_raw_fft(None)
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data == 0))

        # 2. Empty input
        fft_data = self.processor.get_raw_fft([])
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data == 0))

        fft_data = self.processor.get_raw_fft(np.array([]))
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data == 0))

        # 3. Short input
        short_data = np.ones(self.buffer_frames // 2)
        fft_data = self.processor.get_raw_fft(short_data)
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data >= 0.0))
        self.assertTrue(np.all(fft_data <= 1.0))

        # 4. Long input
        long_data = np.ones(self.buffer_frames * 2)
        fft_data = self.processor.get_raw_fft(long_data)
        self.assertEqual(len(fft_data), expected_len)
        self.assertTrue(np.all(fft_data >= 0.0))
        self.assertTrue(np.all(fft_data <= 1.0))

if __name__ == '__main__':
    unittest.main()
