import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
import time
import sys

# --- 1. Configuration ---
FS = 16000
MIC_DIST = 0.08      # 8 cm
SPEED_SOUND = 343.0
MAX_TAU = MIC_DIST / SPEED_SOUND

# FFT Parameters: High N_FFT for accurate Delay Estimation
N_FFT = 2048   
HOP_LEN = 512

INPUT_FILE = "/home/cse-sdpl/paarth/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/samples/mixture_3_sources.wav" 
OUTPUT_FILE = "/home/cse-sdpl/paarth/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/samples/duet_target_90deg.wav"

# --- 2. Core Math ---

def stft_transform(y):
    f, t, Zxx = scipy.signal.stft(y, fs=FS, window='hann', nperseg=N_FFT, noverlap=HOP_LEN)
    return f, t, Zxx

def estimate_parameters(X1, X2, f_freqs):
    """
    [cite_start]Standard DUET estimation of alpha and delta[cite: 10].
    """
    epsilon = 1e-12
    magnitude_threshold = 1e-5 
    energy_mask = (np.abs(X1) > magnitude_threshold) & (np.abs(X2) > magnitude_threshold)
    
    # Ratio X2/X1
    R = (X2[energy_mask] + epsilon) / (X1[energy_mask] + epsilon)
    
    # [cite_start]Delay Estimation (delta = -phase / omega) [cite: 9]
    freq_grid = np.tile(f_freqs[:, np.newaxis], (1, X1.shape[1]))
    w_grid = 2 * np.pi * freq_grid
    w_selected = w_grid[energy_mask]
    
    phase_diff = np.imag(np.log(R))
    delta_est_flat = -phase_diff / (w_selected + epsilon)
    
    # Map back to full TF grid (fill noise with nan or 0)
    delta_full = np.zeros_like(X1, dtype=float)
    delta_full[energy_mask] = delta_est_flat
    
    return delta_full, energy_mask

def extract_broadside_wiener(X1, delta_grid, valid_mask, beam_width_sec=0.00005):
    """
    Extracts ONLY the source at Delay=0 using a smooth Wiener Filter.
    
    beam_width_sec: Standard Deviation of the Gaussian beam.
                    0.00005s is approx +/- 15 degrees width.
    """
    print(f"Extraction Beam Width: {beam_width_sec} seconds")
    epsilon = 1e-12
    
    # 1. Spatial Probability Mask (Gaussian centered at 0)
    # This defines "How likely is this pixel to be the Target?"
    # P(target) = exp( -delta^2 / 2*sigma^2 )
    
    # We only compute this where energy is valid
    spatial_mask = np.zeros_like(delta_grid)
    spatial_mask[valid_mask] = np.exp(-(delta_grid[valid_mask]**2) / (2 * beam_width_sec**2))
    
    # 2. Power Estimation (The Wiener Step)
    # We use the spatial mask to estimate the Power of Signal vs Interference
    P_mixture = np.abs(X1)**2
    
    P_target_est = P_mixture * (spatial_mask**2)
    P_noise_est  = P_mixture * ((1 - spatial_mask)**2)
    
    # 3. Wiener Gain Calculation
    # G = P_target / (P_target + P_noise)
    # This is smooth (0.0 to 1.0) and preserves texture better than Binary Masking
    G_wiener = P_target_est / (P_target_est + P_noise_est + epsilon)
    
    # 4. Apply Filter
    S_hat_freq = X1 * G_wiener
    
    # 5. Inverse Transform
    _, s_hat_time = scipy.signal.istft(S_hat_freq, fs=FS, window='hann', nperseg=N_FFT, noverlap=HOP_LEN)
    
    return s_hat_time

# --- 3. Main ---

if __name__ == "__main__":
    print(f"--- DUET Broadside Target Extractor (90-deg Only) ---")
    
    # Load
    try:
        mixture, fs = sf.read(INPUT_FILE)
        if len(mixture.shape) == 2: mixture = mixture.T 
        x1, x2 = mixture[0], mixture[1]
    except FileNotFoundError:
        print("Please run world.py first to generate the mixture.")
        sys.exit()

    start_time = time.time()

    # Transform
    f, t, X1 = stft_transform(x1)
    f, t, X2 = stft_transform(x2)
    
    # Estimate Delays
    delta_grid, valid_mask = estimate_parameters(X1, X2, f)
    
    # Extract Target (0 Delay)
    # TUNING KNOB: beam_width_sec
    # 0.00003 = Very Sharp (High rejection, more artifacts)
    # 0.00010 = Very Wide (Low rejection, smooth audio)
    target_audio = extract_broadside_wiener(X1, delta_grid, valid_mask, beam_width_sec=0.00006)
    
    # Normalize & Save
    target_audio = target_audio / np.max(np.abs(target_audio))
    sf.write(OUTPUT_FILE, target_audio, FS)
    
    print(f"Processing Complete in {time.time() - start_time:.4f}s")
    print(f"Saved: {OUTPUT_FILE}")