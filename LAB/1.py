import librosa
import librosa.display
import matplotlib.pyplot as plt

def visualize_audio_librosa(audio_path):
    # Load audio file with librosa
    audio_data, sampling_rate = librosa.load(audio_path, sr=None)

    # Create waveform plot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sampling_rate)
    plt.title('Waveform of Audio File')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

# Usage example:
visualize_audio_librosa('madira.flac')