
# Speech and Audio Signal Processing — Labs (ECECE68)

This repository contains lab exercises aligned with the ECECE68 Speech and Audio Signal Processing course. Each experiment includes learning objectives, theory background, tasks, deliverables, and starter guidance for both Python and MATLAB.

Instructor: Dr. Amit Singhal
Chairperson: Dr. Tarun K. Rawat

## Table of Contents

- Prerequisites and Setup
- Repository Structure
- How to Work on Labs
- Experiments

1. Visualizing Audio Signals
2. Frequency-Domain Characterization
3. Generating a Monotone Audio Signal
4. Feature Extraction from Speech
5. Recognition of Spoken Words (Isolated Words)
6. Speech-to-Text (ASR)
7. ARIMA Modeling for Time-Series Forecasting
8. MFCC Implementation
9. Modified Discrete Cosine Transform (MDCT)
10. Mini Project (ML/DL on Speech/Audio)
- Reporting and Submission
- Academic Integrity
- References

***

## Prerequisites and Setup

You may complete labs in Python or MATLAB.

Python (recommended):

- Python 3.9+ (Anaconda recommended)
- Required packages:
    - numpy, scipy, matplotlib, librosa, soundfile
    - scikit-learn, pandas
    - statsmodels (for ARIMA)
    - torch/torchaudio or TensorFlow/Keras (for DL in Exp. 10; optional)
    - jiwer (for WER metrics; optional)
- Installation example:
    - pip install numpy scipy matplotlib librosa soundfile scikit-learn pandas statsmodels jiwer
    - Optional: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

MATLAB:

- Audio Toolbox (recommended)
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox (for some parts)
- Econometrics Toolbox (for ARIMA)

Audio formats:

- Use WAV (mono, 16-bit PCM) where possible.
- Suggested sampling rates: 8 kHz (telephony), 16 kHz (speech), 44.1/48 kHz (music).

***

## Repository Structure

- data/
    - raw/ (original audio)
    - processed/ (cleaned, resampled, normalized)
- notebooks/ (optional exploratory work)
- src/
    - python/
        - exp01_visualize_audio.py
        - exp02_frequency_domain.py
        - exp03_generate_tone.py
        - exp04_feature_extraction.py
        - exp05_isolated_word_recognition.py
        - exp06_asr_stt.py
        - exp07_arima_timeseries.py
        - exp08_mfcc.py
        - exp09_mdct.py
        - exp10_miniproject/
    - matlab/
        - exp01_visualize_audio.m
        - exp02_frequency_domain.m
        - ...
- reports/
    - exp01/
    - exp02/
    - ...
- README.md (this file)

Note: You can adapt filenames as needed; keep each experiment self-contained.

***

## How to Work on Labs

- Create a new branch per experiment (recommended).
- Maintain a clear README or docstrings in each script explaining usage.
- Provide command-line interfaces where possible (e.g., python src/python/exp01_visualize_audio.py --input data/raw/sample.wav).
- Save output plots/figures to reports/expXX/ with descriptive names.
- Include short written answers in a PDF/Markdown report per experiment.

***

## Experiments

### 1) Visualizing Audio Signals — Reading from a File and Working on It

Objectives:

- Read audio from file.
- Plot waveform, amplitude histogram, and spectrogram.
- Normalize audio and trim silence.

Theory:

- Discrete-time audio, sampling, quantization.
- Time-amplitude visualization and dynamic range.

Tasks:

- Load a WAV file (mono preferred).
- Plot:
    - Waveform vs. time
    - Histogram of samples
    - Spectrogram (STFT magnitude)
- Normalize peak to 0.9, optionally trim leading/trailing silence.
- Save processed audio to data/processed/.

Deliverables:

- Plots (PNG)
- Brief report: sampling rate, duration, normalization method.
- Processed audio file.

Python hints:

- librosa.load, soundfile.write, matplotlib.pyplot.specgram or librosa.display.specshow, numpy.histogram.

MATLAB hints:

- audioread, soundsc, spectrogram, histogram.

***

### 2) Frequency-Domain Characterization — Transforming to Frequency Domain

Objectives:

- Compute spectrum via FFT.
- Compare window types and sizes.
- Estimate power spectral density (PSD).

Theory:

- DFT/FFT, windowing, leakage.
- PSD estimation (Welch).

Tasks:

- Implement magnitude spectrum and log-magnitude spectrum.
- Test Hanning, Hamming, Blackman; frame sizes (e.g., 256, 512, 1024).
- Compute PSD via Welch.
- Comment on frequency resolution vs. time resolution trade-offs.

Deliverables:

- Spectrum plots across windows/sizes.
- PSD comparison plot.
- Short discussion on leakage and resolution.

Python hints:

- numpy.fft.rfft, scipy.signal.get_window, scipy.signal.welch.

MATLAB hints:

- fft, periodogram/pwelch, window().

***

### 3) Generating a Monotone Audio Signal

Objectives:

- Synthesize a pure tone with amplitude envelope.
- Export tone to WAV.

Theory:

- Sinusoidal synthesis, amplitude modulation, ADSR envelopes.
- Aliasing considerations.

Tasks:

- Generate sine tone at f0 (e.g., 440 Hz), Fs ∈ {16k, 44.1k}.
- Apply fade-in/fade-out or ADSR envelope.
- Optionally add white noise at specific SNR.

Deliverables:

- WAV file.
- Plot of waveform and spectrogram.
- Write-up on aliasing and envelope parameters.

Python hints:

- numpy.sin, numpy.linspace, soundfile.write.

MATLAB hints:

- sin, linspace, audiowrite.

***

### 4) Feature Extraction from Speech

Objectives:

- Implement common features: STE, ZCR, ACF, spectral centroid, bandwidth, roll-off, MFCCs (preview for Exp. 8).
- Frame-level processing and delta features.

Theory:

- Short-time analysis; common low-level descriptors for speech processing.

Tasks:

- Frame the signal (e.g., 25 ms window, 10 ms hop).
- Compute:
    - Short-Time Energy (STE)
    - Zero-Crossing Rate (ZCR)
    - ACF-based pitch estimate (optional)
    - Spectral centroid, bandwidth, roll-off
- Aggregate features to CSV for a dataset.

Deliverables:

- Feature plots over time.
- CSV of features.
- Brief discussion on feature behavior across voiced/unvoiced/silence.

Python hints:

- librosa.feature functions; custom STE/ZCR via numpy.

MATLAB hints:

- buffer/framing; custom functions; audioFeatureExtractor.

***

### 5) Recognition of Spoken Words (Isolated Word Recognition)

Objectives:

- Build a simple classifier for isolated words (e.g., digits).
- Evaluate accuracy.

Theory:

- Template matching (DTW) or classical ML (k-NN/SVM) with MFCC features.

Tasks:

- Collect small dataset (5–10 classes, 5–10 samples each).
- Extract MFCC or simple features.
- Train classifier (k-NN/SVM) and evaluate with train/test split.
- Optionally DTW-based template matching baseline.

Deliverables:

- Confusion matrix and accuracy.
- Short report comparing methods/features.

Python hints:

- scikit-learn (SVC, KNeighborsClassifier), librosa.feature.mfcc, dtw-python (optional).

MATLAB hints:

- fitcsvm, classificationLearner, mfcc().

***

### 6) Speech Recognition — Converting Speech to Text (ASR)

Objectives:

- Implement a basic ASR pipeline using a pre-trained model or a simple HMM/GMM toy system.
- Evaluate transcription quality.

Theory:

- Acoustic modeling, language modeling basics.
- Modern end-to-end ASR (CTC/Attention/Transducer).

Tasks (two paths):

- Practical path (recommended): Use a lightweight pre-trained ASR (e.g., Vosk, wav2vec2 via torchaudio, HuggingFace pipelines) to transcribe audio samples.
- Educational path: Implement a toy GMM-HMM or CTC decoding for a very limited vocabulary (optional/advanced).

Evaluation:

- Compute WER/CER on a small test set with reference transcripts.

Deliverables:

- Transcriptions (TXT/JSON).
- WER/CER results and observations on noise/sampling rate effects.

Python hints:

- Vosk API, transformers + torchaudio; jiwer for WER.

MATLAB hints:

- Limited built-ins for ASR; focus on feature extraction + simple template matching or external integration.

***

### 7) ARIMA Model for Time Series Forecasting

Note: Included in Unit-IV lab list; applicable to audio-derived time series (e.g., energy contour, F0 contour).

Objectives:

- Fit ARIMA to a derived time series from audio.
- Forecast short horizons and evaluate.

Theory:

- ARIMA(p,d,q) modeling, stationarity, ACF/PACF.

Tasks:

- Extract a time series from audio (e.g., frame-wise STE or pitch).
- Split into train/test; fit ARIMA using AIC/BIC for model selection.
- Forecast on test and compute MAE/RMSE.
- Discuss suitability of ARIMA for speech-derived series.

Deliverables:

- ACF/PACF plots; forecast vs. ground truth plot.
- Error metrics and short discussion.

Python hints:

- statsmodels.tsa.arima.model.ARIMA.

MATLAB hints:

- arima, estimate, forecast.

***

### 8) Mel-Frequency Cepstral Coefficients (MFCC)

Objectives:

- Implement MFCC extraction pipeline.
- Compare against library outputs.

Theory:

- Pre-emphasis, framing, windowing.
- STFT magnitude, Mel filterbank, log, DCT to cepstra.
- Delta and delta-delta.

Tasks:

- Implement MFCC step-by-step:
    - Pre-emphasis (optional)
    - Framing (e.g., 25 ms, 10 ms hop), Hamming window
    - STFT magnitude
    - Mel filterbank, log energies
    - DCT-II to get cepstra (e.g., 13 coefficients)
    - Compute delta and delta-delta
- Validate by comparing with librosa.mfcc or MATLAB mfcc.

Deliverables:

- MFCC heatmap (time vs. coefficient).
- Comparison metrics (L2 diff) with library output.
- Brief write-up of design choices.

Python hints:

- librosa.filters.mel, scipy.fftpack.dct (or numpy), librosa.feature.mfcc for reference.

MATLAB hints:

- mfcc, designAuditoryFilterBank for custom.

***

### 9) Modified Discrete Cosine Transform (MDCT)

Objectives:

- Implement MDCT and inverse MDCT with appropriate window (e.g., sine window) and 50% overlap.
- Explore time/frequency tiling.

Theory:

- MDCT, lapped transforms, time-domain aliasing cancellation (TDAC).
- Basis for AAC.

Tasks:

- Implement MDCT and IMDCT with overlap-add.
- Test perfect reconstruction on test signals (tone, speech).
- Inspect pre-echo artifacts with transients.

Deliverables:

- Reconstruction error plot.
- Spectral illustrations around transients.
- Discussion on windowing and TDAC.

Python hints:

- Implement from definition; use numpy for matrix ops; sine window; frame length N, hop N/2.

MATLAB hints:

- Custom MDCT/IMDCT; dsp.MDCT (if available) or custom functions.

***

### 10) Mini Project — ML/DL on Speech/Audio

Objectives:

- Design and implement a small end-to-end project applying ML/DL to speech/audio.

Example topics:

- Keyword spotting (wake-word detection)
- Speaker verification or identification
- Noise reduction/speech enhancement (spectral subtraction, Wiener, DNN)
- Emotion recognition from speech
- Music genre classification
- Language identification (LID)

Expected components:

- Problem statement and related work.
- Data acquisition and preprocessing.
- Feature engineering or end-to-end pipeline.
- Model training, validation, and testing.
- Metrics and error analysis.
- Reproducibility (seed, environment, instructions).
- Ethical considerations and limitations.

Deliverables:

- Code with README to run.
- Report (4–8 pages) with figures and results.
- Optional short demo video or notebook.

***

## Reporting and Submission

For each experiment:

- Code: Place scripts/notebooks under src/python or src/matlab.
- Data: Maintain small samples under data/ (do not commit large proprietary datasets).
- Outputs: Place plots, tables, and audio outputs under reports/expXX/.
- Report: Include a concise write-up (PDF or Markdown) covering:
    - Objectives
    - Methods and parameters
    - Results (figures/tables)
    - Discussion and conclusions
    - Challenges and future work

Evaluation alignment:

- TCA and PCA may include vivas/lab tests; keep your pipeline reproducible.
- Lab files and performance are assessed each turn—commit regularly with clear messages.

***

## Academic Integrity

- You may use libraries for standard functions but must understand and explain your pipeline.
- Cite all external code or pretrained models used.
- Plagiarism or uncredited reuse will affect assessment.

***

## References

- Ben Gold, Nelson Morgan, Speech and Audio Signal Processing: Processing and Perception of Speech and Music.
- L. R. Rabiner, R. W. Schafer, Digital Processing of Speech Signals.
- A. R. Jayan, Speech and Audio Signal Processing.
- Khalid Sayood, Introduction to Data Compression.

***

## Quickstart Commands (Python examples)

- Visualize audio:
    - python src/python/exp01_visualize_audio.py --input data/raw/sample.wav --out reports/exp01/
- Frequency domain:
    - python src/python/exp02_frequency_domain.py --input data/raw/sample.wav --win hamming --nfft 1024
- Generate tone:
    - python src/python/exp03_generate_tone.py --f0 440 --fs 16000 --dur 2.0 --snr 30
- Features:
    - python src/python/exp04_feature_extraction.py --input data/raw/sample.wav --csv reports/exp04/features.csv
- Isolated word recognition:
    - python src/python/exp05_isolated_word_recognition.py --data data/words/ --out reports/exp05/
- STT:
    - python src/python/exp06_asr_stt.py --input data/raw/speech.wav --out reports/exp06/transcript.txt
- ARIMA:
    - python src/python/exp07_arima_timeseries.py --series ste --input data/raw/sample.wav
- MFCC:
    - python src/python/exp08_mfcc.py --input data/raw/sample.wav --compare-librosa
- MDCT:
    - python src/python/exp09_mdct.py --input data/raw/sample.wav --N 1024
- Mini project:
    - See src/python/exp10_miniproject/README.md for instructions.

***

Notes:

- Sampling rate consistency matters; resample if needed.
- Always check for clipping after normalization/synthesis.
- Keep random seeds fixed for reproducibility where relevant.

<div style="text-align: center">⁂</div>


