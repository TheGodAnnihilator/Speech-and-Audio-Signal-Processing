import librosa
import numpy as np

def extract_mfcc(audio_path, n_mfcc=13):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Take the mean of each MFCC across all frames (to represent the full audio with a simple vector)
    mfcc_mean = np.mean(mfcc_features, axis=1)
    return mfcc_mean

# Usage example:
features = extract_mfcc('madira.flac')
print(features)

