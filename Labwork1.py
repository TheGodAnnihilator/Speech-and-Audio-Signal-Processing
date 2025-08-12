"""
Audio Visualizer with File Picker

1.  Lets you browse for an audio file (.wav, .mp3, .flac, …).
2.  Prints detailed metadata (duration, sample-rate, channels, dtype, bit-depth).
3.  Shows:
      • First 10 raw samples in the console
      • Full-length waveform in a Matplotlib window
----------------------------------------------------------------------
Required third-party packages
----------------------------------------------------------------------
pip install soundfile   # reliable multi-format reader
pip install librosa     # convenient metadata helpers
pip install matplotlib  # plotting
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import soundfile as sf          # reads many audio formats
import librosa                  # lightweight helper for duration in s
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def choose_file() -> str:
    """Open a native file-browser dialog and return the chosen path."""
    root = tk.Tk()
    root.withdraw()                     # hide the empty root window
    filetypes = [
        ("Audio files", "*.wav *.flac *.mp3 *.ogg *.aiff *.aif *.m4a"),
        ("All files",   "*.*")
    ]
    path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=filetypes
    )
    if not path:
        messagebox.showinfo("No selection", "No file was chosen.")
        sys.exit(0)
    return path

def print_metadata(file_path: str, y: np.ndarray, sr: int):
    """Collect and display human-readable metadata."""
    # Generic information
    duration = len(y) / sr
    channels = 1 if y.ndim == 1 else y.shape[1]
    bit_depth = y.dtype.itemsize * 8
    file_size = os.path.getsize(file_path) / (1024**2)  # MiB

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

def plot_waveform(y: np.ndarray, sr: int, file_name: str):
    """Display a time-domain waveform."""
    t = np.linspace(0, len(y)/sr, len(y))
    plt.figure(figsize=(12, 5))
    plt.plot(t, y, color="steelblue", linewidth=0.6)
    plt.title(f"Waveform – {file_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    path = choose_file()                                # 1 Select
    y, sr = sf.read(path, always_2d=False)              # 2 Load audio data
    if y.ndim > 1:                                      # convert stereo→mono
        y = y.mean(axis=1)
    print_metadata(path, y, sr)                         # 3 Metadata + raw view
    plot_waveform(y, sr, os.path.basename(path))        # 4 Visualization

if __name__ == "__main__":
    main()
