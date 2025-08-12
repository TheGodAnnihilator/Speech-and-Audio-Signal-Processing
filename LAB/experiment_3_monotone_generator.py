"""
Experiment 3: Monotone Audio Signal Generation
Based on the Speech and Audio Signal Processing course lecture plan

Functionality:
1. Generates monotone (single frequency) audio signals
2. Allows user to specify frequency, duration, amplitude, and waveform type
3. Saves generated audio to file
4. Visualizes the generated signal in time and frequency domains
5. Supports multiple waveform types: sine, square, sawtooth, triangle
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import os
from scipy.signal import square, sawtooth
from scipy.fft import fft, fftfreq

class MonotoneGenerator:
    def __init__(self):
        self.sample_rate = 44100  # Standard audio sample rate
        self.generated_signal = None
        self.frequency = None
        self.duration = None
        
    def get_user_parameters(self):
        """Get signal parameters from user input."""
        root = tk.Tk()
        root.withdraw()
        
        # Get frequency
        freq = simpledialog.askfloat(
            "Frequency Input",
            "Enter frequency in Hz (20-20000):",
            initialvalue=440.0,
            minvalue=20.0,
            maxvalue=20000.0
        )
        if freq is None:
            messagebox.showinfo("Cancelled", "Signal generation cancelled.")
            return False
        
        # Get duration
        dur = simpledialog.askfloat(
            "Duration Input",
            "Enter duration in seconds (0.1-30):",
            initialvalue=3.0,
            minvalue=0.1,
            maxvalue=30.0
        )
        if dur is None:
            messagebox.showinfo("Cancelled", "Signal generation cancelled.")
            return False
        
        # Get amplitude
        amp = simpledialog.askfloat(
            "Amplitude Input",
            "Enter amplitude (0.1-1.0):",
            initialvalue=0.7,
            minvalue=0.1,
            maxvalue=1.0
        )
        if amp is None:
            messagebox.showinfo("Cancelled", "Signal generation cancelled.")
            return False
        
        # Get waveform type
        waveform = simpledialog.askstring(
            "Waveform Type",
            "Enter waveform type (sine/square/sawtooth/triangle):",
            initialvalue="sine"
        )
        if waveform is None:
            messagebox.showinfo("Cancelled", "Signal generation cancelled.")
            return False
        
        waveform = waveform.lower().strip()
        if waveform not in ['sine', 'square', 'sawtooth', 'triangle']:
            messagebox.showerror("Error", "Invalid waveform type. Using sine wave.")
            waveform = 'sine'
        
        self.frequency = freq
        self.duration = dur
        self.amplitude = amp
        self.waveform_type = waveform
        
        return True
    
    def generate_monotone_signal(self):
        """Generate the monotone audio signal based on user parameters."""
        # Time array
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        
        # Generate waveform based on type
        if self.waveform_type == 'sine':
            signal = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        elif self.waveform_type == 'square':
            signal = self.amplitude * square(2 * np.pi * self.frequency * t)
        elif self.waveform_type == 'sawtooth':
            signal = self.amplitude * sawtooth(2 * np.pi * self.frequency * t)
        elif self.waveform_type == 'triangle':
            signal = self.amplitude * sawtooth(2 * np.pi * self.frequency * t, width=0.5)
        else:
            signal = self.amplitude * np.sin(2 * np.pi * self.frequency * t)  # Default to sine
        
        # Apply fade-in and fade-out to avoid clicks
        fade_samples = int(0.01 * self.sample_rate)  # 10ms fade
        if len(signal) > 2 * fade_samples:
            # Fade-in
            signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # Fade-out
            signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        self.generated_signal = signal
        self.time_array = t
        
        return signal, t
    
    def analyze_generated_signal(self):
        """Analyze the generated signal in time and frequency domains."""
        if self.generated_signal is None:
            print("No signal generated yet!")
            return
        
        # Time domain analysis
        print("\n==========  GENERATED SIGNAL ANALYSIS  ==========")
        print(f"Waveform type:    {self.waveform_type.capitalize()}")
        print(f"Frequency:        {self.frequency:.1f} Hz")
        print(f"Duration:         {self.duration:.2f} s")
        print(f"Amplitude:        {self.amplitude:.2f}")
        print(f"Sample rate:      {self.sample_rate} Hz")
        print(f"Total samples:    {len(self.generated_signal)}")
        print(f"RMS value:        {np.sqrt(np.mean(self.generated_signal**2)):.4f}")
        print(f"Peak amplitude:   {np.max(np.abs(self.generated_signal)):.4f}")
        
        # Frequency domain analysis
        Y = fft(self.generated_signal)
        freqs = fftfreq(len(self.generated_signal), 1/self.sample_rate)
        
        # Positive frequencies only
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        Y_pos = Y[pos_mask]
        magnitude = np.abs(Y_pos)
        
        # Find peak frequency
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs_pos[peak_idx]
        print(f"Peak frequency:   {peak_freq:.1f} Hz")
        print("==================================================\n")
    
    def visualize_signal(self):
        """Create comprehensive visualization of the generated signal."""
        if self.generated_signal is None:
            print("No signal to visualize!")
            return
        
        # FFT for frequency domain
        Y = fft(self.generated_signal)
        freqs = fftfreq(len(self.generated_signal), 1/self.sample_rate)
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        magnitude = np.abs(Y[pos_mask])
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Full time domain signal
        ax1.plot(self.time_array, self.generated_signal, color="blue", linewidth=1)
        ax1.set_title(f"{self.waveform_type.capitalize()} Wave - {self.frequency}Hz", fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, self.duration)
        
        # Plot 2: Zoomed view of first few cycles
        cycles_to_show = min(5, self.frequency * self.duration)
        time_zoom = cycles_to_show / self.frequency
        zoom_samples = int(time_zoom * self.sample_rate)
        ax2.plot(self.time_array[:zoom_samples], self.generated_signal[:zoom_samples], 
                color="red", linewidth=2, marker='o', markersize=2)
        ax2.set_title(f"Zoomed View - First {cycles_to_show:.1f} Cycles", fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(alpha=0.3)
        
        # Plot 3: Frequency spectrum
        ax3.plot(freqs_pos[:len(freqs_pos)//2], magnitude_db[:len(freqs_pos)//2], 
                color="green", linewidth=1)
        ax3.set_title("Magnitude Spectrum", fontweight="bold")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude (dB)")
        ax3.grid(alpha=0.3)
        ax3.set_xlim(0, min(2000, self.sample_rate//2))  # Focus on lower frequencies
        
        # Plot 4: Frequency spectrum around fundamental (zoomed)
        freq_range = max(100, self.frequency * 0.5)  # Show range around fundamental
        freq_mask = (freqs_pos >= self.frequency - freq_range) & (freqs_pos <= self.frequency + freq_range)
        ax4.plot(freqs_pos[freq_mask], magnitude_db[freq_mask], 
                color="purple", linewidth=2, marker='o', markersize=3)
        ax4.axvline(x=self.frequency, color='red', linestyle='--', alpha=0.7, 
                   label=f'Fundamental: {self.frequency}Hz')
        ax4.set_title("Spectrum Around Fundamental Frequency", fontweight="bold")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Magnitude (dB)")
        ax4.grid(alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_audio_file(self):
        """Save the generated signal to an audio file."""
        if self.generated_signal is None:
            print("No signal to save!")
            return
        
        root = tk.Tk()
        root.withdraw()
        
        # Default filename
        default_name = f"monotone_{self.waveform_type}_{int(self.frequency)}Hz_{self.duration}s.wav"
        
        filepath = filedialog.asksaveasfilename(
            title="Save generated audio",
            defaultextension=".wav",
            initialvalue=default_name,
            filetypes=[
                ("WAV files", "*.wav"),
                ("FLAC files", "*.flac"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            # Ensure signal is in proper range for audio file
            normalized_signal = self.generated_signal / np.max(np.abs(self.generated_signal))
            sf.write(filepath, normalized_signal, self.sample_rate)
            print(f"Audio saved to: {filepath}")
            messagebox.showinfo("Success", f"Audio saved successfully to:\n{os.path.basename(filepath)}")
        else:
            print("Save cancelled by user.")

def main():
    """Main function implementing Experiment 3: Monotone Audio Signal Generation"""
    print("EXPERIMENT 3: MONOTONE AUDIO SIGNAL GENERATION")
    print("==============================================")
    
    generator = MonotoneGenerator()
    
    # Step 1: Get user parameters
    if not generator.get_user_parameters():
        return
    
    print(f"\nGenerating {generator.waveform_type} wave:")
    print(f"- Frequency: {generator.frequency} Hz")
    print(f"- Duration: {generator.duration} s")
    print(f"- Amplitude: {generator.amplitude}")
    
    # Step 2: Generate the signal
    signal, time_array = generator.generate_monotone_signal()
    print("Signal generation completed!")
    
    # Step 3: Analyze the signal
    generator.analyze_generated_signal()
    
    # Step 4: Visualize the signal
    print("Displaying signal visualization...")
    generator.visualize_signal()
    
    # Step 5: Option to save
    save_choice = messagebox.askyesno("Save Audio", "Do you want to save the generated audio to a file?")
    if save_choice:
        generator.save_audio_file()
    
    print("\nExperiment 3 completed successfully!")
    print("Generated monotone signal with:")
    print("- Time domain visualization")
    print("- Frequency domain analysis")
    print("- Optional audio file export")

if __name__ == "__main__":
    main()