import librosa
import numpy as np
import matplotlib.pyplot as plt

def show_frequency_domain(audio_path):
    # Load audio file
    audio_data, sampling_rate = librosa.load(audio_path, sr=None)

    # Compute FFT (frequency domain)
    fft_output = np.fft.fft(audio_data)
    fft_magnitude = np.abs(fft_output)

    # Get the frequencies corresponding to FFT values
    freqs = np.fft.fftfreq(len(fft_magnitude), 1/sampling_rate)

    # Only plot the positive half of frequencies
    positives = freqs > 0

    plt.figure(figsize=(10, 4))
    plt.plot(freqs[positives], fft_magnitude[positives])
    plt.title('Frequency Domain of Audio Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()

# Usage example:
show_frequency_domain('madira.flac')
