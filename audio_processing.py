"""
Audio processing module for calculating FFT and frequency binning.
"""
import numpy as np

# FFT Calculation Constants
DB_CONVERSION_COEFFICIENT = 20
LOG_EPSILON = 1e-6

# Normalization Constants
# _calculate_fft_magnitude divides the raw FFT magnitude by the window's
# coherent gain, so magnitudes are expressed in units of input amplitude and
# are independent of buffer_frames: a full-scale DC signal reads 1.0 (0 dBFS)
# and a full-scale sine reads 0.5 (-6 dBFS) for any transform length.
# We therefore display a fixed 60 dB window: [-60 dBFS, 0 dBFS] -> [0.0, 1.0].
NORMALIZATION_OFFSET = 60
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

        # Coherent gain of the analysis window. Dividing the FFT magnitude by
        # it makes the dB scale independent of buffer_frames (see constants).
        self._window_gain = float(np.sum(self.window)) or 1.0

        # Frequencies of the rfft output bins. Depends only on the sample rate
        # and buffer size, so it is computed once instead of on every frame.
        self.freqs = np.fft.rfftfreq(buffer_frames, 1.0 / sample_rate)

        # Cache of log-spaced band edges, keyed by (num_bars, min_freq, max_freq).
        self._band_cache = {}

    def set_sample_rate(self, sample_rate):
        """
        Retune the analysis to a new sample rate.

        A file being played back sets the rate, not the capture device, so the
        bin frequencies (and the bands derived from them) have to be rebuilt
        when the source changes. No-op when the rate is unchanged, so callers
        can set it unconditionally.
        """
        sample_rate = int(sample_rate)
        if sample_rate <= 0 or sample_rate == self.sample_rate:
            return
        self.sample_rate = sample_rate
        self.freqs = np.fft.rfftfreq(self.buffer_frames, 1.0 / sample_rate)
        # Band edges are derived from self.freqs and from sample_rate / 2.
        self._band_cache.clear()

    def _calculate_fft_magnitude(self, audio_data):
        """
        Helper method to compute FFT magnitude in dB.
        Handles padding/truncation and windowing.
        """
        audio_data = np.asarray(audio_data, dtype=float)

        # Sanitize input: replace NaN/Inf with 0
        if not np.all(np.isfinite(audio_data)):
            audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure audio data matches buffer size (pad if necessary).
        # Short buffers are padded at the end and long ones truncated from the
        # front, so the newest samples are always kept.
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

        # Scale by the window's coherent gain so the result is in units of
        # input amplitude rather than growing linearly with buffer_frames.
        fft_mag = np.abs(fft_complex) / self._window_gain

        # Convert to dBFS
        fft_mag = DB_CONVERSION_COEFFICIENT * np.log10(fft_mag + LOG_EPSILON)

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

        return self._bin_frequencies(fft_mag, self.freqs, num_bars)

    def _band_indices(self, num_bars):
        """
        Return (starts, ends, fallback) index arrays for `num_bars` log-spaced
        bands, computed once per configuration and cached.

        `starts`/`ends` slice the FFT output for each band; `fallback` holds the
        index of the FFT bin closest to each band's center, used when a band is
        narrower than the FFT resolution.
        """
        key = (num_bars, self.min_freq, self.max_freq)
        cached = self._band_cache.get(key)
        if cached is not None:
            return cached

        # We use a logarithmic scale for frequency banding since human hearing
        # is logarithmic. Avoid zero or negative frequency for log scale.
        min_f = max(self.min_freq, 1)
        max_f = min(self.max_freq, self.sample_rate / 2)
        log_freq_edges = np.logspace(np.log10(min_f), np.log10(max_f), num_bars + 1)

        # self.freqs is sorted ascending, so searchsorted yields the same band
        # membership as (freqs >= start) & (freqs < end) in O(log n) per edge.
        starts = np.searchsorted(self.freqs, log_freq_edges[:-1], side='left')
        ends = np.searchsorted(self.freqs, log_freq_edges[1:], side='left')

        # Closest bin to each band center, for bands with no FFT bin of their own.
        centers = (log_freq_edges[:-1] + log_freq_edges[1:]) / 2.0
        pos = np.clip(np.searchsorted(self.freqs, centers), 1, len(self.freqs) - 1)
        left, right = self.freqs[pos - 1], self.freqs[pos]
        fallback = np.where(centers - left <= right - centers, pos - 1, pos)

        result = (starts, ends, fallback)
        self._band_cache[key] = result
        return result

    def _bin_frequencies(self, fft_mag, freqs, num_bars):
        starts, ends, fallback = self._band_indices(num_bars)

        binned_values = np.empty(num_bars)
        for i in range(num_bars):
            start, end = starts[i], ends[i]
            if end > start:
                # Use max magnitude in this band to ensure narrow peaks (like
                # sine waves) are visible.
                binned_values[i] = fft_mag[start:end].max()
            else:
                # Fallback: If the frequency band is narrower than our FFT
                # resolution (common for low frequencies with many bars), just
                # grab the closest bin.
                binned_values[i] = fft_mag[fallback[i]]

        # Normalize and clamp output to 0.0 - 1.0 range.
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
