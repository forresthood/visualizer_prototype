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
        # remain consistent (regression testing). The values are pinned, not just
        # checked for positivity, so a change to the dB scale is caught here.

        # Use a constant input
        audio_data = np.full(self.buffer_frames, 0.1)

        # A DC level of 0.1 is -20 dBFS, which the [-60, 0] dB display window
        # maps to 40/60 of full scale.
        fft_data = self.processor.get_raw_fft(audio_data)
        self.assertAlmostEqual(fft_data[0], 40.0 / 60.0, places=4)

        # The lowest bar spans 20-21 Hz, narrower than the 43 Hz bin spacing, so
        # it falls back to the closest bin: the window's first sidelobe at half
        # the DC amplitude (-26 dBFS).
        bars = self.processor.compute_fft(audio_data, 32)
        self.assertAlmostEqual(bars[0], 0.56653814, places=6)

    def test_scale_is_independent_of_buffer_size(self):
        """
        The same signal must produce the same bar heights at any buffer size.

        FFT magnitude grows linearly with the transform length, so without
        normalizing by the window's coherent gain the fixed NORMALIZATION_OFFSET
        and NORMALIZATION_SCALE calibration drifts with buffer_frames -- the
        tests here (1024) would validate a different amplitude scale than
        main.py (2048) actually renders.
        """
        peaks = []
        for buffer_frames in (512, 1024, 2048, 4096):
            processor = AudioProcessor(sample_rate=self.sample_rate,
                                       buffer_frames=buffer_frames)
            t = np.arange(buffer_frames) / self.sample_rate
            tone = 0.1 * np.sin(2 * np.pi * 1000 * t)
            peaks.append(np.max(processor.get_raw_fft(tone)))

        # Remaining spread is scalloping loss from where the tone falls relative
        # to bin centers, not a change of scale.
        self.assertLess(max(peaks) - min(peaks), 0.05,
                        f"peak level varies with buffer size: {peaks}")
        for peak in peaks:
            self.assertGreater(peak, 0.4)
            self.assertLess(peak, 0.7)

    def test_full_scale_reference_levels(self):
        """A full-scale DC signal is 0 dBFS; a full-scale sine is -6 dBFS."""
        dc = self.processor.get_raw_fft(np.ones(self.buffer_frames))
        self.assertAlmostEqual(dc[0], 1.0, places=6)

        t = np.arange(self.buffer_frames) / self.sample_rate
        sine = self.processor.get_raw_fft(np.sin(2 * np.pi * 1000 * t))
        # -6 dBFS in a [-60, 0] dB window is 54/60 of full scale.
        self.assertAlmostEqual(np.max(sine), 54.0 / 60.0, delta=0.02)


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


class TestSampleRateRetuning(unittest.TestCase):
    """
    A file being played back sets the sample rate, not the capture device, so
    the analysis has to follow it.
    """

    def setUp(self):
        self.processor = AudioProcessor(sample_rate=44100, buffer_frames=2048)

    def test_bin_frequencies_follow_the_new_rate(self):
        self.processor.set_sample_rate(48000)

        self.assertEqual(self.processor.sample_rate, 48000)
        self.assertAlmostEqual(self.processor.freqs[-1], 24000.0)
        np.testing.assert_allclose(
            self.processor.freqs, np.fft.rfftfreq(2048, 1.0 / 48000))

    def test_band_cache_is_rebuilt_for_the_new_rate(self):
        """Stale band edges would map every bar to the wrong frequencies."""
        self.processor.compute_fft(np.zeros(2048), 32)
        self.assertTrue(self.processor._band_cache)

        self.processor.set_sample_rate(8000)
        self.assertFalse(self.processor._band_cache)

    def test_a_tone_reads_at_its_true_frequency_at_any_rate(self):
        """
        A 1 kHz tone must peak in the band that spans 1 kHz whichever rate the
        file was recorded at. Which band that is differs between rates, since
        the log-spaced bands stop at Nyquist; the frequency it stands for
        must not.
        """
        num_bars = 32
        for rate in (22050, 44100, 48000):
            with self.subTest(rate=rate):
                processor = AudioProcessor(sample_rate=44100, buffer_frames=2048)
                processor.set_sample_rate(rate)

                t = np.arange(2048) / rate
                bins = processor.compute_fft(np.sin(2 * np.pi * 1000 * t), num_bars)
                peak = int(np.argmax(bins))

                edges = np.logspace(np.log10(processor.min_freq),
                                    np.log10(min(processor.max_freq, rate / 2)),
                                    num_bars + 1)
                self.assertLessEqual(edges[peak], 1000.0)
                self.assertGreaterEqual(edges[peak + 1], 1000.0)

    def test_unchanged_and_invalid_rates_are_ignored(self):
        self.processor.compute_fft(np.zeros(2048), 32)
        cached = dict(self.processor._band_cache)

        for rate in (44100, 0, -1):
            with self.subTest(rate=rate):
                self.processor.set_sample_rate(rate)
                self.assertEqual(self.processor.sample_rate, 44100)
                # An ignored call must not throw the cache away either.
                self.assertEqual(list(self.processor._band_cache), list(cached))


if __name__ == '__main__':
    unittest.main()
