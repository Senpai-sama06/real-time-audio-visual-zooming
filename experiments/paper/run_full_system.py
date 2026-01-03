import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from scipy.io import wavfile
import sys
import os

from config import SystemConfig
from classifier import SpatioSpectralClassifier
from actuator import SpatioSpectralActuator

def load_audio(filename, target_fs):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        sys.exit(1)
    fs, audio = wavfile.read(filename)
    if fs != target_fs:
        print(f"Warning: File is {fs}Hz, System is {target_fs}Hz.")
        # In a real app, resample here. For now, warn.
    
    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
        
    return audio

if __name__ == "__main__":
    # --- SETUP ---
    INPUT_FILE = "mixture.wav" # Ensure this exists
    OUTPUT_FILE = "enhanced_output.wav"
    
    cfg = SystemConfig()
    
    # 1. Initialize Modules
    classifier = SpatioSpectralClassifier(target_angle_rad=1.57)
    actuator = SpatioSpectralActuator(cfg)
    
    # 2. Load Data
    print(f"Loading {INPUT_FILE}...")
    audio = load_audio(INPUT_FILE, cfg.fs)
    
    # STFT parameters
    n_fft = cfg.n_fft
    hop = n_fft // 4 # Standard 25% overlap
    
    print("Computing STFT...")
    f, t, Zxx = stft(audio[:,0], fs=cfg.fs, nperseg=n_fft, noverlap=hop)
    _, _, Zyy = stft(audio[:,1], fs=cfg.fs, nperseg=n_fft, noverlap=hop)
    Y_stft = np.stack([Zxx, Zyy], axis=-1) # [Bins, Frames, 2]
    
    n_bins, n_frames, _ = Y_stft.shape
    
    # Output buffer
    S_hat_stft = np.zeros((n_bins, n_frames), dtype=complex)
    
    print(f"Processing {n_frames} frames...")
    
    # 3. REAL-TIME LOOP SIMULATION
    for i in range(n_frames):
        # A. Grab current frame
        Y_t = Y_stft[:, i, :] # [Bins, 2]
        
        # B. ESTIMATOR: Get Hypothesis & State
        # Note: classifier updates its internal noise/covariance state automatically
        decisions = classifier.process_frame(Y_t)
        
        # Retrieve the updated Noise Covariance from the classifier's feature extractor
        # (The actuator needs this for the Benesty filter)
        # We construct R_nn from the classifier's internal noise floor estimate for simplicity
        # or use the full R_smooth if the classifier flagged it as H0/H6.
        # Here, we use a hybrid approach:
        # Use the Classifier's smoothed Covariance (R_smooth) as the "Phi" basis
        R_current = classifier.features.R_smooth
        
        # C. ACTUATOR: Resolve Spectrum
        S_hat_stft[:, i] = actuator.process_frame(Y_t, R_current, decisions)
        
        if i % 100 == 0:
            print(f"\rProgress: {i}/{n_frames}", end="")
            
    print("\nInverse STFT...")
    _, s_time = istft(S_hat_stft, fs=cfg.fs, nperseg=n_fft, noverlap=hop)
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(s_time))
    if max_val > 0:
        s_time = s_time / max_val * 0.9
        
    print(f"Saving to {OUTPUT_FILE}...")
    wavfile.write(OUTPUT_FILE, cfg.fs, (s_time * 32767).astype(np.int16))
    
    # 4. PLOTTING
    print("Generating Plots...")
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Input Spectrogram
    ax[0].pcolormesh(t, f, 10*np.log10(np.abs(Zxx)**2 + 1e-12), cmap='inferno', shading='auto')
    ax[0].set_title("Input (Noisy)")
    ax[0].set_ylabel("Freq [Hz]")
    
    # Output Spectrogram
    ax[1].pcolormesh(t, f, 10*np.log10(np.abs(S_hat_stft)**2 + 1e-12), cmap='inferno', shading='auto')
    ax[1].set_title("Output (Enhanced)")
    ax[1].set_ylabel("Freq [Hz]")
    
    # Waveforms
    ax[2].plot(np.linspace(0, len(audio)/cfg.fs, len(audio)), audio[:,0], alpha=0.5, label="Input")
    ax[2].plot(np.linspace(0, len(s_time)/cfg.fs, len(s_time)), s_time, alpha=0.8, color='k', label="Output")
    ax[2].set_title("Waveform Overlay")
    ax[2].set_xlabel("Time [s]")
    ax[2].legend()
    
    plt.tight_layout()
    plt.show()