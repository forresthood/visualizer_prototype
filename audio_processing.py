import numpy as np

class AudioProcessor:
    def __init__(self, sample_rate=44100, buffer_frames=1024):
        self.sample_rate = sample_rate
        self.buffer_frames = buffer_frames
        self.window = np.hanning(buffer_frames)
        self.min_freq = 20
        self.max_freq = 20000

    def compute_fft(self, audio_data, num_bars):
        if audio_data is None or len(audio_data) == 0:
            return np.zeros(num_bars)
            
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
        fft_mag = 20 * np.log10(fft_mag + 1e-6) # Convert to dB scale
        
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
                # Average magnitude in this band
                binned_values[i] = np.mean(fft_mag[idx])
            else:
                # Fallback: If the frequency band is narrower than our FFT resolution 
                # (common for low frequencies with many bars), just grab the closest bin.
                center_freq = (start_freq + end_freq) / 2.0
                closest_idx = np.argmin(np.abs(freqs - center_freq))
                binned_values[i] = fft_mag[closest_idx]
                
        # Normalize and clamp output to 0.0 - 1.0 range
        # dB range is roughly -120 to 0 (scaled differently based on input amplitude)
        # We map roughly from [-40 dB, 40 dB] down to [0, 1] for visualization
        normalized = (binned_values + 20) / 60.0
        clamped = np.clip(normalized, 0.0, 1.0)
            
        return clamped

    def get_raw_fft(self, audio_data):
        """Return the full FFT magnitude array normalized to 0-1, for spectrogram use."""
        if audio_data is None or len(audio_data) == 0:
            return np.zeros(self.buffer_frames // 2 + 1)
            
        # Ensure audio data matches buffer size
        if len(audio_data) < self.buffer_frames:
            padded_data = np.zeros(self.buffer_frames)
            padded_data[:len(audio_data)] = audio_data
            audio_data = padded_data
        elif len(audio_data) > self.buffer_frames:
            audio_data = audio_data[-self.buffer_frames:]

        windowed_data = audio_data * self.window
        fft_complex = np.fft.rfft(windowed_data)
        fft_mag = np.abs(fft_complex)
        fft_mag = 20 * np.log10(fft_mag + 1e-6)
        
        # Normalize to 0-1
        normalized = (fft_mag + 20) / 60.0
        return np.clip(normalized, 0.0, 1.0)
