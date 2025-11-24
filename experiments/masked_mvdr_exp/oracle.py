import numpy as np
import scipy.signal
import soundfile as sf
import os
import argparse
import sys
from profiler_utils import profile_performance

# --- 1. Constants (Matched to oracle_debug.py) ---
FS = 16000
N_MICS = 2
N_FFT = 1024
N_HOP = 512
D = 0.08  
C = 343.0
ANGLE_TARGET = 90.0
# Using 1e-6 for diagonal loading (standard robust MVDR) 
# instead of the '1' found in the debug script, which is too aggressive.
SIGMA = 1e-6 

# --- 2. Physics Helper ---

def get_steering_vector(angle_deg, f, d, c):
    theta_rad = np.deg2rad(angle_deg)
    # Simple far-field model (Elevation = 0)
    tau_m1 = (d / 2) * np.cos(0) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(0) * np.cos(theta_rad - np.pi) / c
    omega = 2 * np.pi * f
    d_vec = np.array([[np.exp(-1j * omega * tau_m1)], [np.exp(-1j * omega * tau_m2)]], dtype=complex)
    return d_vec

# --- 3. Pipeline Execution ---

def run_oracle_pipeline(mix_path, ref_tgt_path, ref_int_path, output_path):
    # 1. Load Files
    try:
        y_mix, _ = sf.read(mix_path, dtype='float32')
        s_tgt, _ = sf.read(ref_tgt_path, dtype='float32')
        s_int, _ = sf.read(ref_int_path, dtype='float32')
    except Exception as e:
        print(f"[Oracle] Error loading files: {e}")
        sys.exit(1)

    # Ensure shape is (Channels, Time)
    if y_mix.ndim > 1: y_mix = y_mix.T 
    
    # 2. Compute STFTs
    f, t, Y_mix = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_tgt = scipy.signal.stft(s_tgt, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_int = scipy.signal.stft(s_int, fs=FS, nperseg=N_FFT, noverlap=N_HOP)

    # 3. Construct the ORACLE MASK (The Cheat)
    # Mask = 1 if Interference Energy > Target Energy
    mag_tgt = np.abs(S_tgt)
    mag_int = np.abs(S_int)
    mask_noise = np.where(mag_int > mag_tgt, 1.0, 0.0)
    
    # 4. MVDR Processing
    n_freqs, n_frames = Y_mix.shape[1], Y_mix.shape[2]
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    
    # Pre-compute steering vectors to save time inside the loop
    d_vectors = [get_steering_vector(ANGLE_TARGET, f[i], D, C) for i in range(n_freqs)]

    for f_idx in range(n_freqs):
        # Skip low frequencies as per original script
        if f[f_idx] < 100: 
            continue 

        m_f = mask_noise[f_idx, :] 
        Y_f = Y_mix[:, f_idx, :] 
        
        # Estimate Noise Covariance (R_n) using the Oracle Mask
        # Weighted by the mask (only look at noise-dominant bins)
        Y_weighted = Y_f * np.sqrt(m_f) 
        R_noise = (Y_weighted @ Y_weighted.conj().T) / (np.sum(m_f) + 1e-6)
        
        # Diagonal Loading for stability
        R = R_noise + SIGMA * np.eye(N_MICS)
        d = d_vectors[f_idx]
        
        # Calculate Beamformer Weights: w = R^-1 * d / (d^H * R^-1 * d)
        try:
            R_inv_d = np.linalg.solve(R, d)
            w = R_inv_d / (d.conj().T @ R_inv_d + 1e-12)
        except np.linalg.LinAlgError:
            # Fallback to delay-and-sum or just mic1 if matrix is singular
            w = np.array([[1], [0]])

        # Apply Weights
        S_mvdr[f_idx, :] = w.conj().T @ Y_f

    # 5. Apply Aggressive Post-Filtering
    # In the original script, we multiply the MVDR output by the inverse mask
    # 1.0 = Target Dominant, 0.0 = Noise Dominant
    mask_target = 1.0 - mask_noise 
    S_final = S_mvdr * mask_target
    
    # 6. Reconstruction
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    # Normalize to avoid clipping
    s_out /= (np.max(np.abs(s_out)) + 1e-9)
    
    sf.write(output_path, s_out, FS)

# --- 4. Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Pipeline A: Oracle MVDR (Theoretical Upper Bound)")
    parser.add_argument("--mix_path", type=str, required=True, help="Path to mixture wav")
    parser.add_argument("--ref_tgt", type=str, required=True, help="Path to target reference wav (for cheating)")
    parser.add_argument("--ref_int", type=str, required=True, help="Path to interference reference wav (for cheating)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output wav")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to save stats json")
    args = parser.parse_args()

    # Wrap execution in profiler
    with profile_performance(args.stats_path):
        run_oracle_pipeline(args.mix_path, args.ref_tgt, args.ref_int, args.output_path)

if __name__ == "__main__":
    main()