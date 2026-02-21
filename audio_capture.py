import soundcard as sc
import numpy as sc_np
import threading
import queue

class AudioCaptureSubsystem:
    def __init__(self, sample_rate=44100, buffer_frames=1024):
        self.sample_rate = sample_rate
        self.buffer_frames = buffer_frames
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.capture_thread = None

    def start(self):
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()

    def _capture_loop(self):
        try:
            # Using default speaker in loopback mode
            default_speaker = sc.default_speaker()
            mics = sc.all_microphones(include_loopback=True)
            
            # Find the loopback microphone for the default speaker
            loopback_mic = None
            for mic in mics:
                if mic.isloopback and default_speaker.id in mic.id:
                    loopback_mic = mic
                    break
            
            if not loopback_mic:
                # Fallback to the default loopback if specific one not found
                for mic in mics:
                    if mic.isloopback and default_speaker.name in mic.name:
                        loopback_mic = mic
                        break
            
            if not loopback_mic:
                 # Absolute fallback: just get the first loopback
                 for mic in mics:
                     if mic.isloopback:
                         loopback_mic = mic
                         break
            
            print(f"Using default speaker: {default_speaker.name}")
            print(f"Selected loopback mic: {loopback_mic.name if loopback_mic else 'None'}")
                        
            with loopback_mic.recorder(samplerate=self.sample_rate) as mic:
                while self.is_running:
                    # Record a chunk of audio
                    data = mic.record(numframes=self.buffer_frames)
                    
                    # Convert to mono by averaging channels
                    mono_data = data.mean(axis=1)
                    
                    # Put in queue, discard oldest if full to avoid lag
                    try:
                        self.audio_queue.put_nowait(mono_data)
                    except queue.Full:
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(mono_data)
                        except (queue.Empty, queue.Full):
                            pass
        except Exception as e:
            print(f"Error in audio capture: {e}")
            
    def get_latest_data(self):
        latest_data = None
        # Drain the queue to get the most recent chunk
        while not self.audio_queue.empty():
            try:
                latest_data = self.audio_queue.get_nowait()
            except queue.Empty:
                break
        return latest_data
