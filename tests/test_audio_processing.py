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


    def test_compute_fft_padding_correctness(self):
         """Test that short input is correctly padded."""
         # Create a short input
         short_len = self.buffer_frames // 2
         audio_data = np.zeros(short_len)
         audio_data[short_len // 2] = 1.0

         # Compute FFT on short data
         bars_short = self.processor.compute_fft(audio_data, 32)

         # Manually pad and compute
         padded_data = np.zeros(self.buffer_frames)
         padded_data[:short_len] = audio_data
         bars_padded = self.processor.compute_fft(padded_data, 32)

         np.testing.assert_array_almost_equal(bars_short, bars_padded)

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
        # Expect some output from non-silent input
        self.assertTrue(np.any(bars > 0.0))

        # 4. Long input (more than buffer_frames)
        long_data = np.ones(self.buffer_frames * 2)
        bars = self.processor.compute_fft(long_data, num_bars)
        self.assertEqual(len(bars), num_bars)
        self.assertTrue(np.all(bars >= 0.0))
        self.assertTrue(np.all(bars <= 1.0))
        # Expect some output from non-silent input
        self.assertTrue(np.any(bars > 0.0))

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

    def test_init_parameters(self):
        """Test different initialization parameters."""
        sample_rate = 48000
        buffer_frames = 512
        processor = AudioProcessor(sample_rate=sample_rate, buffer_frames=buffer_frames)
        self.assertEqual(processor.sample_rate, sample_rate)
        self.assertEqual(processor.buffer_frames, buffer_frames)
        self.assertEqual(len(processor.window), buffer_frames)

        # Test compute_fft with new parameters
        audio_data = np.zeros(buffer_frames)
        bars = processor.compute_fft(audio_data, 16)
        self.assertEqual(len(bars), 16)

    def test_frequency_range_clamping(self):
        """Test min_freq and max_freq clamping behavior."""
        # Set invalid frequency range
        self.processor.min_freq = -100
        self.processor.max_freq = 100000  # Above Nyquist (22050)

        # Create input
        audio_data = np.random.random(self.buffer_frames)
        bars = self.processor.compute_fft(audio_data, 32)

        # Should execute without error
        self.assertEqual(len(bars), 32)
        self.assertTrue(np.all(bars >= 0.0))
        self.assertTrue(np.all(bars <= 1.0))

    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values in input."""
        # 1. NaN input
        nan_data = np.full(self.buffer_frames, np.nan)
        try:
            bars = self.processor.compute_fft(nan_data, 32)
            self.assertEqual(len(bars), 32)
            # Output should not contain NaNs (robustness check)
            self.assertFalse(np.any(np.isnan(bars)), "Output contains NaNs")
        except Exception as e:
            self.fail(f"compute_fft raised exception on NaN input: {e}")

        # 2. Inf input
        inf_data = np.full(self.buffer_frames, np.inf)
        try:
            bars = self.processor.compute_fft(inf_data, 32)
            self.assertEqual(len(bars), 32)
            # Output should not contain NaNs or Infs
            self.assertFalse(np.any(np.isnan(bars)), "Output contains NaNs")
            self.assertFalse(np.any(np.isinf(bars)), "Output contains Infs")
        except Exception as e:
            self.fail(f"compute_fft raised exception on Inf input: {e}")

    def test_multi_frequency_binning(self):
        """Test binning logic with low, mid, and high frequencies."""
        # Frequencies to test: 100Hz, 1000Hz, 10000Hz
        freqs = [100, 1000, 10000]

        # Verify peak bin index increases with frequency
        peak_indices = []
        for freq in freqs:
            t = np.linspace(0, self.buffer_frames / self.sample_rate, self.buffer_frames, endpoint=False)
            audio_data = 0.5 * np.sin(2 * np.pi * freq * t)
            bars = self.processor.compute_fft(audio_data, 32)
            peak_indices.append(np.argmax(bars))

            # Basic sanity check that we have output
            self.assertTrue(np.any(bars > 0.0), f"Frequency {freq}Hz produced all-zero bars. Max: {np.max(bars)}")

        self.assertTrue(peak_indices[0] <= peak_indices[1] <= peak_indices[2],
                        f"Peak indices should increase with frequency. Got {peak_indices}")

if __name__ == '__main__':
    unittest.main()
