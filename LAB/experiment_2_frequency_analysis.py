"""
Experiment 2: Audio Signal Frequency Domain Analysis
Based on the Speech and Audio Signal Processing course lecture plan

Functionality:
1. Loads audio file using file picker
2. Displays metadata and raw samples
3. Performs frequency domain transformation using FFT
4. Shows magnitude spectrum, phase spectrum, and spectrogram
5. Analyzes frequency characteristics of the audio signal
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq

def choose_file() -> str:
    """Open a native file-browser dialog and return the chosen path."""
    root = tk.Tk()
    root.withdraw()
    filetypes = [
        ("Audio files", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif *.m4a"),
        ("All files", "*.*")
    ]
    path = filedialog.askopenfilename(
        title="Select an audio file for frequency analysis",
        filetypes=filetypes
    )
    if not path:
        messagebox.showinfo("No selection", "No file was chosen.")
        sys.exit(0)
    return path

def print_metadata(file_path: str, y: np.ndarray, sr: int):
    """Display comprehensive audio metadata."""
    duration = len(y) / sr
    channels = 1 if y.ndim == 1 else y.shape[1]
    bit_depth = y.dtype.itemsize * 8
    file_size = os.path.getsize(file_path) / (1024**2)
    
    print("\n==========  AUDIO METADATA  ==========")
    print(f"File:           {os.path.basename(file_path)}")
    print(f"File format:    {sf.info(file_path).format}")
    print(f"Duration:       {duration:.2f} s")
    print(f"Sample rate:    {sr} Hz")
    print(f"Channels:       {channels}")
    print(f"Data type:      {y.dtype}  ({bit_depth}-bit)")
    print(f"File size:      {file_size:.2f} MiB")
    print("First 10 raw samples:", y[:10])
    print("======================================\n")

def analyze_frequency_domain(y: np.ndarray, sr: int, file_name: str):
    """
    Perform comprehensive frequency domain analysis of the audio signal.
    This implements the core requirement of Experiment 2.
    """
    # 1. Compute FFT
    n_samples = len(y)
    Y = fft(y)
    freqs = fftfreq(n_samples, 1/sr)
    
    # Take only positive frequencies
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    Y_pos = Y[pos_mask]
    
    # Magnitude and phase spectra
    magnitude = np.abs(Y_pos)
    phase = np.angle(Y_pos)
    
    # Convert to dB scale for better visualization
    magnitude_db = 20 * np.log10(magnitude + 1e-12)  # avoid log(0)
    
    # Create comprehensive frequency domain plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time domain waveform (reference)
    t = np.linspace(0, len(y)/sr, len(y))
    ax1.plot(t, y, color="steelblue", linewidth=0.6)
    ax1.set_title(f"Time Domain - {file_name}", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    
    # Plot 2: Magnitude Spectrum
    ax2.plot(freqs_pos[:len(freqs_pos)//2], magnitude_db[:len(freqs_pos)//2], 
             color="red", linewidth=0.8)
    ax2.set_title("Magnitude Spectrum", fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, sr//2)  # Nyquist frequency limit
    
    # Plot 3: Phase Spectrum
    ax3.plot(freqs_pos[:len(freqs_pos)//2], phase[:len(freqs_pos)//2], 
             color="green", linewidth=0.8)
    ax3.set_title("Phase Spectrum", fontweight="bold")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Phase (radians)")
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, sr//2)
    
    # Plot 4: Spectrogram (time-frequency representation)
    f_spec, t_spec, Sxx = spectrogram(y, sr, nperseg=1024, noverlap=512)
    im = ax4.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), 
                        shading='gouraud', cmap='viridis')
    ax4.set_title("Spectrogram", fontweight="bold")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax4, label='Power (dB)')
    
    plt.tight_layout()
    plt.show()
    
    # Print frequency domain analysis results
    print("==========  FREQUENCY DOMAIN ANALYSIS  ==========")
    print(f"Frequency resolution: {sr/n_samples:.2f} Hz")
    print(f"Nyquist frequency: {sr/2} Hz")
    
    # Find dominant frequencies
    dominant_freq_indices = np.argsort(magnitude)[-5:][::-1]  # Top 5
    print("\nTop 5 Dominant Frequencies:")
    for i, idx in enumerate(dominant_freq_indices):
        freq = freqs_pos[idx]
        mag_db = magnitude_db[idx]
        print(f"{i+1}. {freq:.1f} Hz ({mag_db:.1f} dB)")
    
    # Spectral centroid (brightness measure)
    spectral_centroid = np.sum(freqs_pos * magnitude) / np.sum(magnitude)
    print(f"\nSpectral Centroid: {spectral_centroid:.1f} Hz")
    
    # Bandwidth calculation
    spectral_spread = np.sqrt(np.sum(((freqs_pos - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
    print(f"Spectral Spread: {spectral_spread:.1f} Hz")
    print("=================================================\n")

def main():
    """Main function implementing Experiment 2: Frequency Domain Analysis"""
    print("EXPERIMENT 2: AUDIO SIGNAL FREQUENCY DOMAIN ANALYSIS")
    print("====================================================")
    
    # Step 1: Select audio file
    file_path = choose_file()
    
    # Step 2: Load and preprocess audio
    y, sr = sf.read(file_path, always_2d=False)
    if y.ndim > 1:  # Convert stereo to mono
        y = y.mean(axis=1)
    
    # Step 3: Display metadata
    print_metadata(file_path, y, sr)
    
    # Step 4: Perform frequency domain analysis
    analyze_frequency_domain(y, sr, os.path.basename(file_path))
    
    print("Frequency domain analysis completed!")
    print("The analysis shows:")
    print("- Time domain waveform")
    print("- Magnitude spectrum (frequency content)")
    print("- Phase spectrum (phase relationships)")
    print("- Spectrogram (time-frequency evolution)")

if __name__ == "__main__":
    main()