import numpy as np
import scipy.signal
import soundfile as sf
import argparse
import sys
import os

# Import the profiler we created in Step 1
from profiler_utils import profile_performance

# --- 1. Configuration (Preserved from Original) ---
FS = 16000
MIC_DIST = 0.08      
SPEED_SOUND = 343.0
MAX_TAU = MIC_DIST / SPEED_SOUND

# Using a slightly larger FFT to help distinguish close sources
N_FFT = 4096   
HOP_LEN = 1024

# --- 2. Core Math & Helpers ---

def stft_transform(y):
    # Hann window with 75% overlap
    f, t, Zxx = scipy.signal.stft(y, fs=FS, window='hann', nperseg=N_FFT, noverlap=HOP_LEN)
    return f, t, Zxx

def estimate_estimates(X1, X2, f_freqs):
    epsilon = 1e-12
    magnitude_threshold = 1e-5 
    
    # 1. Energy Mask
    energy_mask = (np.abs(X1) > magnitude_threshold) & (np.abs(X2) > magnitude_threshold)
    
    # 2. Ratio & Delay
    R = (X2[energy_mask] + epsilon) / (X1[energy_mask] + epsilon)
    a_flat = np.abs(R)
    
    # Frequency mapping
    freq_grid = np.tile(f_freqs[:, np.newaxis], (1, X1.shape[1]))
    w_grid = 2 * np.pi * freq_grid
    w_selected = w_grid[energy_mask]
    
    # Phase & Delay
    phase_diff = np.imag(np.log(R))
    delta_flat = -phase_diff / (w_selected + epsilon)
    
    return delta_flat, a_flat, energy_mask

def find_interference_peaks(delta_flat, target_delay=0.0):
    """
    Scans the histogram to find the strongest INTERFERENCE sources 
    ignoring the Target (at 0.0).
    """
    # Create a 1D Histogram of delays
    hist, bins = np.histogram(delta_flat, bins=200, range=(-MAX_TAU*1.5, MAX_TAU*1.5))
    centers = (bins[:-1] + bins[1:]) / 2
    
    # Smooth the histogram to find robust peaks
    hist_smooth = scipy.signal.savgol_filter(hist, window_length=11, polyorder=3)
    
    # Find Peaks
    peak_idxs, props = scipy.signal.find_peaks(hist_smooth, height=np.max(hist_smooth)*0.1, distance=10)
    found_delays = centers[peak_idxs]
    
    # Filter out the Target (which is near 0)
    interference_delays = []
    for d in found_delays:
        if abs(d - target_delay) > 0.00003: # If peak is more than 30us away from target
            interference_delays.append(d)
            
    # Fallback if no peaks found (e.g., quiet or diffuse noise)
    if not interference_delays:
        # Default fallback to approx 40 and 130 deg
        return [0.00015, -0.00015] 
        
    return interference_delays

def competitive_wiener_filter(X1, delta_full, valid_mask, target_delay, interf_delays, sigma=0.00005):
    """
    Creates a mask where sources COMPETE for energy.
    Mask_Target = P(Target) / (P(Target) + P(Int_A) + P(Int_B) + Noise)
    """
    # Initialize Probability Maps
    p_target = np.zeros_like(delta_full)
    p_interference = np.zeros_like(delta_full)
    
    # 1. Calculate Unnormalized Gaussian Probabilities
    d_vals = delta_full[valid_mask]
    
    # Score for Target (Centered at 0.0)
    score_tgt = np.exp(-(d_vals - target_delay)**2 / (2 * sigma**2))
    p_target[valid_mask] = score_tgt
    
    # Score for Interferers (Sum of probabilities of all found interferers)
    score_int_total = np.zeros_like(score_tgt)
    for i_delay in interf_delays:
        score_i = np.exp(-(d_vals - i_delay)**2 / (2 * sigma**2))
        score_int_total += score_i
        
    p_interference[valid_mask] = score_int_total
    
    # 2. Competitive Gain Calculation
    noise_floor = 1e-9 
    
    # G = P_T / (P_T + P_I + N)
    G_competitive = p_target / (p_target + p_interference + noise_floor)
    
    # 3. Apply Filter
    S_hat_freq = X1 * G_competitive
    
    # 4. Inverse Transform
    _, s_hat_time = scipy.signal.istft(S_hat_freq, fs=FS, window='hann', nperseg=N_FFT, noverlap=HOP_LEN)
    
    return s_hat_time

# --- 3. Pipeline Execution ---

def run_duet_pipeline(mix_path, output_path):
    # 1. Load Data
    try:
        mix, fs = sf.read(mix_path)
        if len(mix.shape) == 2: mix = mix.T # Ensure shape (2, Samples)
        x1, x2 = mix[0], mix[1]
    except Exception as e:
        print(f"[DUET] Error loading file {mix_path}: {e}")
        sys.exit(1)
        
    # 2. STFT
    f, t, X1 = stft_transform(x1)
    f, t, X2 = stft_transform(x2)
    
    # 3. Estimates
    delta_flat, a_flat, valid_mask = estimate_estimates(X1, X2, f)
    
    # Create a full-grid delay map for masking
    delta_grid = np.zeros_like(X1, dtype=float)
    delta_grid[valid_mask] = delta_flat
    
    # 4. Find The Enemy (Interferers)
    # We assume Target is at 0. We find the other peaks.
    interf_delays = find_interference_peaks(delta_flat, target_delay=0.0)
    
    # 5. Competitive Separation
    # sigma controls selectivity. 0.00004 is approx 13us.
    target_audio = competitive_wiener_filter(X1, delta_grid, valid_mask, 
                                           target_delay=0.0, 
                                           interf_delays=interf_delays, 
                                           sigma=0.00004)
    
    # 6. Save
    # Normalize to prevent clipping
    target_audio = target_audio / (np.max(np.abs(target_audio)) + 1e-9)
    sf.write(output_path, target_audio, FS)

# --- 4. Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Pipeline B: DUET Blind Source Separation")
    parser.add_argument("--mix_path", type=str, required=True, help="Path to input mixture wav")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output wav")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to save timing/RAM stats json")
    args = parser.parse_args()

    # Wrap the execution in the profiler to measure Time and Peak RAM
    with profile_performance(args.stats_path):
        run_duet_pipeline(args.mix_path, args.output_path)

if __name__ == "__main__":
    main()