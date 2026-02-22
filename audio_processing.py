"""
Audio processing module for calculating FFT and frequency binning.
"""
import numpy as np

# FFT Calculation Constants
DB_CONVERSION_COEFFICIENT = 20
LOG_EPSILON = 1e-6

# Normalization Constants
NORMALIZATION_OFFSET = 20
NORMALIZATION_SCALE = 60.0
MIN_NORMALIZED_VALUE = 0.0
MAX_NORMALIZED_VALUE = 1.0

class AudioProcessor:
    """
    Processes audio data to compute FFT and frequency bins for visualization.
    """
    def __init__(self, sample_rate=44100, buffer_frames=1024):
        self.sample_rate = sample_rate
        self.buffer_frames = buffer_frames
        self.window = np.hanning(buffer_frames)
        self.min_freq = 20
        self.max_freq = 20000

    def _calculate_fft_magnitude(self, audio_data):
        """
        Helper method to compute FFT magnitude in dB.
        Handles padding/truncation and windowing.
        """
        # Sanitize input: replace NaN/Inf with 0
        if not np.all(np.isfinite(audio_data)):
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure audio data matches buffer size (pad if necessary)
        if len(audio_data) < self.buffer_frames:
            padded_data = np.zeros(self.buffer_frames)
            padded_data[:len(audio_data)] = audio_data
            audio_data = padded_data
        elif len(audio_data) > self.buffer_frames:
            audio_data = audio_data[-self.buffer_frames:]

        # Apply Hanning window to reduce spectral leakage
        windowed_data = audio_data * self.window

        # Compute FFT
        fft_complex = np.fft.rfft(windowed_data)

        # Calculate magnitudes and normalize
        fft_mag = np.abs(fft_complex)
        fft_mag = DB_CONVERSION_COEFFICIENT * np.log10(fft_mag + LOG_EPSILON) # Convert to dB scale

        return fft_mag

    def compute_fft(self, audio_data, num_bars):
        """
        Computes the FFT of the audio data and bins it into `num_bars`.

        Args:
            audio_data: The input audio data array.
            num_bars: The number of frequency bars to produce.

        Returns:
            A numpy array of length `num_bars` with normalized values between 0.0 and 1.0.
        """
        if audio_data is None or len(audio_data) == 0:
            return np.zeros(num_bars)

        fft_mag = self._calculate_fft_magnitude(audio_data)

        # Get frequencies corresponding to FFT bins
        freqs = np.fft.rfftfreq(self.buffer_frames, 1.0 / self.sample_rate)

        return self._bin_frequencies(fft_mag, freqs, num_bars)

    def _bin_frequencies(self, fft_mag, freqs, num_bars):
        # We use a logarithmic scale for frequency banding since human hearing is logarithmic
        # Avoid zero or negative frequency for log scale
        min_f = max(self.min_freq, 1)
        max_f = min(self.max_freq, self.sample_rate / 2)

        log_freq_edges = np.logspace(np.log10(min_f), np.log10(max_f), num_bars + 1)

        binned_values = np.zeros(num_bars)

        for i in range(num_bars):
            start_freq = log_freq_edges[i]
            end_freq = log_freq_edges[i+1]

            # Find indices in the FFT output that fall into this band
            idx = np.where((freqs >= start_freq) & (freqs < end_freq))[0]

            if len(idx) > 0:
                # Use max magnitude in this band to ensure narrow peaks (like sine waves) are visible
                binned_values[i] = np.max(fft_mag[idx])
            else:
                # Fallback: If the frequency band is narrower than our FFT resolution
                # (common for low frequencies with many bars), just grab the closest bin.
                center_freq = (start_freq + end_freq) / 2.0
                closest_idx = np.argmin(np.abs(freqs - center_freq))
                binned_values[i] = fft_mag[closest_idx]

        # Normalize and clamp output to 0.0 - 1.0 range
        # dB range is roughly -120 to 0 (scaled differently based on input amplitude)
        # We map roughly from [-40 dB, 40 dB] down to [0, 1] for visualization
        normalized = (binned_values + NORMALIZATION_OFFSET) / NORMALIZATION_SCALE
        clamped = np.clip(normalized, MIN_NORMALIZED_VALUE, MAX_NORMALIZED_VALUE)

        return clamped

    def get_raw_fft(self, audio_data):
        """Return the full FFT magnitude array normalized to 0-1, for spectrogram use."""
        if audio_data is None or len(audio_data) == 0:
            return np.zeros(self.buffer_frames // 2 + 1)

        fft_mag = self._calculate_fft_magnitude(audio_data)

        # Normalize to 0-1
        normalized = (fft_mag + NORMALIZATION_OFFSET) / NORMALIZATION_SCALE
        return np.clip(normalized, MIN_NORMALIZED_VALUE, MAX_NORMALIZED_VALUE)
