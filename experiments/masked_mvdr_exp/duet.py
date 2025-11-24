import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import tracemalloc
import functools
import sys

# --- 1. Configuration ---
# Must match the settings in your world.py
FS = 16000
MIC_DIST = 0.08      # 8 cm
SPEED_SOUND = 343.0
MAX_TAU = MIC_DIST / SPEED_SOUND

# FFT Parameters
# Trade-off: Larger N_FFT = Better Delay Resolution = Better Separation
# But Larger N_FFT = More Blur in Time = "Pre-echo" artifacts
N_FFT = 2048   
HOP_LEN = 512  # 75% overlap

# Input/Output Paths
# Update this to match where your world.py saved the file
INPUT_FILE = "/home/cse-sdpl/paarth/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/samples/mixture_3_sources.wav" 

# --- 2. Benchmarking Tool ---
def benchmark(func):
    """
    Measures execution time and peak memory usage.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        end_time = time.perf_counter()
        tracemalloc.stop()
        
        time_taken = end_time - start_time
        memory_mb = peak / (1024 * 1024)
        print(f"PERFORMANCE [{func.__name__}]:")
        print(f"   Time:   {time_taken:.4f} seconds")
        print(f"   Memory: {memory_mb:.2f} MB")
        print(f"   -----------------------------")
        return result
    return wrapper

# --- 3. Core Processing ---

@benchmark
def stft_transform(y):
    # Using 'hann' window is standard for speech to minimize leakage
    f, t, Zxx = scipy.signal.stft(y, fs=FS, window='hann', nperseg=N_FFT, noverlap=HOP_LEN)
    return f, t, Zxx

@benchmark
def estimate_parameters(X1, X2, f_freqs):
    """
    Calculates the Ratio (a) and Delay (delta) for every T-F bin.
    """
    epsilon = 1e-12
    
    # 1. Thresholding: Ignore silent bins to reduce noise in the histogram
    # Adjust this threshold if your audio is very quiet
    magnitude_threshold = 1e-5 
    energy_mask = (np.abs(X1) > magnitude_threshold) & (np.abs(X2) > magnitude_threshold)
    
    # 2. Compute Ratio R = X2 / X1
    # We add epsilon to avoid division by zero
    R = (X2[energy_mask] + epsilon) / (X1[energy_mask] + epsilon)
    
    # 3. Amplitude Estimate (a)
    a_est = np.abs(R)
    
    # 4. Delay Estimate (delta)
    # math: delta = -imag(ln(X2/X1)) / omega
    
    # Create a grid of omega (angular freq) values matching the mask
    # Shape of X1 is (n_freq, n_time)
    # f_freqs is (n_freq,)
    freq_grid = np.tile(f_freqs[:, np.newaxis], (1, X1.shape[1]))
    w_grid = 2 * np.pi * freq_grid
    w_selected = w_grid[energy_mask]
    
    # Calculate Phase
    phase_diff = np.imag(np.log(R))
    
    # Calculate Delay
    delta_est = -phase_diff / (w_selected + epsilon)
    
    return a_est, delta_est, energy_mask

def plot_histogram(a_est, delta_est):
    """
    Plots the DUET 2D Histogram.
    """
    print("Generating Histogram Plot...")
    plt.figure(figsize=(12, 8))
    
    # Filter visualization to realistic range
    # Remove crazy outliers caused by division by small frequencies
    valid_mask = (np.abs(delta_est) < MAX_TAU * 1.5) & (a_est < 5) & (a_est > 0.2)
    
    d_plot = delta_est[valid_mask]
    a_plot = a_est[valid_mask]
    
    # Using LogNorm to see smaller clusters
    plt.hist2d(d_plot, a_plot, bins=[150, 150], cmap='inferno', norm=LogNorm())
    plt.colorbar(label='Density (Log Scale)')
    plt.title(f'DUET Histogram ({len(d_plot)} points)')
    plt.xlabel('Delay (seconds)')
    plt.ylabel('Amplitude Ratio (a)')
    
    # Draw reference lines for physics limits
    plt.axvline(MAX_TAU, color='white', linestyle='--', alpha=0.5, label='Max Lag')
    plt.axvline(-MAX_TAU, color='white', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

@benchmark
def duet_demix_soft(X1, X2, a_peaks, delta_peaks, sigma_delta=0.05, sigma_alpha=0.2):
    """
    Demixing with Soft Masking (Gaussian Kernel) to improve audio quality.
    
    sigma_delta: Spread of the mask along the delay axis (tuning parameter)
    sigma_alpha: Spread of the mask along the amplitude axis
    """
    n_freq, n_time = X1.shape
    n_sources = len(a_peaks)
    epsilon = 1e-12
    
    # Re-calculate parameters for the full grid
    R = (X2 + epsilon) / (X1 + epsilon)
    a_grid = np.abs(R)
    
    freq_grid = np.tile(np.linspace(0, FS/2, n_freq)[:, np.newaxis], (1, n_time))
    w_grid = 2 * np.pi * freq_grid
    delta_grid = -np.imag(np.log(R)) / (w_grid + epsilon)
    
    # Calculate Likelihoods (Soft Assignment)
    # We use a Gaussian kernel based on distance to the peak center
    likelihoods = np.zeros((n_sources, n_freq, n_time))
    
    # Normalize delay distance by MAX_TAU so it's comparable to amplitude distance (approx 1.0)
    norm_factor = MAX_TAU 
    
    for i in range(n_sources):
        # Distance metrics
        d_dist = (delta_grid - delta_peaks[i]) / norm_factor
        a_dist = (a_grid - a_peaks[i])
        
        # Gaussian Kernel
        # P(source|pixel) ~ exp(-dist^2 / 2sigma^2)
        score = np.exp(-(d_dist**2)/(2*sigma_delta**2) - (a_dist**2)/(2*sigma_alpha**2))
        likelihoods[i, :, :] = score

    # Normalize likelihoods so they sum to 1 at each pixel (Soft Mask)
    total_likelihood = np.sum(likelihoods, axis=0) + epsilon
    masks = likelihoods / total_likelihood
    
    recovered_signals = []
    
    print("Reconstructing signals...")
    for i in range(n_sources):
        # Apply Soft Mask to Mixture 1 (Reference)
        S_hat_freq = X1 * masks[i]
        
        # ISTFT
        _, s_hat_time = scipy.signal.istft(S_hat_freq, fs=FS, window='hann', nperseg=N_FFT, noverlap=HOP_LEN)
        recovered_signals.append(s_hat_time)
        
    return recovered_signals

# --- 4. Main Execution ---

if __name__ == "__main__":
    print(f"--- DUET Blind Source Separation ---")
    
    # A. Load Audio
    try:
        mixture, fs = sf.read(INPUT_FILE)
        print(f"Loaded {INPUT_FILE} | Shape: {mixture.shape} | SR: {fs}")
        if fs != FS:
            print(f"Warning: Sample rate mismatch. Expected {FS}, got {fs}")
        
        # Ensure Stereo (2, N)
        if mixture.shape[1] == 2:
            mixture = mixture.T
        x1, x2 = mixture[0], mixture[1]
        
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found. Run world.py first.")
        sys.exit()

    # B. Transform
    f, t, X1 = stft_transform(x1)
    f, t, X2 = stft_transform(x2)
    
    # C. Estimate
    a_est, delta_est, _ = estimate_parameters(X1, X2, f)
    
    # D. Visualize
    # You MUST look at this plot to update the `est_delta_peaks` below if the angles change.
    plot_histogram(a_est, delta_est)
    
    # E. Define Peaks (Based on your world.py angles: 90, 40, 130 degrees)
    # 90 deg -> 0 delay
    # 40 deg -> positive delay
    # 130 deg -> negative delay
    # We assume amplitude ratio is ~1.0 for all (since microphones are identical gain)
    
    # --- MANUAL PEAK INPUT ---
    # Look at the histogram X-axis to fine-tune these values!
    # Theoretical values based on d=0.08m:
    # Max delay = 0.08/343 = 0.000233 s
    peak_1_d = 0.0          # Center (90 deg)
    peak_2_d = 0.00015      # Positive (40 deg)
    peak_3_d = -0.00015     # Negative (130 deg)
    
    est_a_peaks = [1.0, 1.0, 1.0] 
    est_delta_peaks = [peak_1_d, peak_2_d, peak_3_d]
    
    print(f"Using Peak Centers (Delay): {est_delta_peaks}")

    # F. Demix (Soft Masking)
    # sigma_delta controls how "strict" the clustering is. 
    # Smaller = more rejection (artifacts), Larger = more bleeding (interference).
    recovered_sources = duet_demix_soft(
        X1, X2, 
        est_a_peaks, est_delta_peaks, 
        sigma_delta=0.2,  # Tuning knob for "Sharpness" of separation
        sigma_alpha=0.5
    )
    
    # G. Save Output
    print("Saving files...")
    for i, sig in enumerate(recovered_sources):
        # Normalize to prevent clipping
        sig = sig / np.max(np.abs(sig))
        out_name = f"duet_recovered_source_{i}.wav"
        sf.write(out_name, sig, FS)
        print(f"   Saved {out_name}")
        
    print("Done.")