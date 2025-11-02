import numpy as np
import sounddevice as sd

def generate_monotone(frequency=440, duration=2, sampling_rate=44100):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    audio_signal = 0.5 * np.sin(2 * np.pi * frequency * t)  # amplitude 0.5 for safety
    sd.play(audio_signal, sampling_rate)
    sd.wait()  # Wait until playback is finished

# Usage example: Play a 440 Hz tone (A4 note) for 2 seconds
generate_monotone(440, 2)
