import librosa
import librosa.display
import matplotlib.pyplot as plt

def compute_mfcc(audio_path, n_mfcc=13):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Compute MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Display MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    return mfccs

# Usage example:
mfcc_features = compute_mfcc('madira.flac')
print(mfcc_features.shape)
