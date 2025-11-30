import numpy as np
import scipy.signal
import soundfile as sf
import os
import argparse
import glob
import sys

# --- Dynamic Configuration Constants ---
BASE_RESULTS_DIR = os.path.join(os.getcwd(), "simulation_results")

# Find the latest timestamped folder automatically for default behavior
def get_latest_run_dir():
    if not os.path.exists(BASE_RESULTS_DIR):
        return "simulation_results/latest_run"
    search_pattern = os.path.join(BASE_RESULTS_DIR, '*_*_*') 
    all_runs = sorted(glob.glob(search_pattern), reverse=True)
    return all_runs[0] if all_runs else "simulation_results/latest_run"

# DEFAULT_OUTDIR = get_latest_run_dir()
DEFAULT_OUTDIR = "simulation_results/ljspeech_reverb_20251130_215709"
# print(DEFAULT_OUTDIR)

# --- Import Core Constants and Helpers ---
try:
    from rt_av_zoom.core.masked_mvdr import (
        get_steering_vector, 
        D, C, N_MICS, FS, N_FFT, N_HOP
    )
except ImportError:
    print("WARNING: Could not import 'rt_av_zoom.core.masked_mvdr'. Using fallback defaults.")
    # Fallback constants
    D = 0.08; C = 343.0; N_MICS = 2; FS = 16000; N_FFT = 512; N_HOP = 256
    def get_steering_vector(angle, f, d, c):
        tau = (d * np.cos(np.deg2rad(angle))) / c
        return np.exp(-1j * 2 * np.pi * f * tau * np.arange(N_MICS))

# --- Experiment Constants ---
ANGLE_TARGET = 90.0

def main(args):
    outdir = args.outdir
    sigma = args.sigma
    hp_cutoff = args.hp
    
    print(f"\n--- ORACLE OPTIMIZATION RUN ---")
    print(f"Directory:  {os.path.basename(outdir)}")
    print(f"Parameters: Sigma={sigma} | HP_Cutoff={hp_cutoff} Hz")
    
    # 1. Validation & Paths
    if not os.path.exists(outdir):
        print(f"ERROR: Directory not found: {outdir}")
        return
        
    path_mix = os.path.join(outdir, "mixture_wpe.wav")
    path_tgt = os.path.join(outdir, "target_reference.wav")
    path_int = os.path.join(outdir, "interference_reference.wav")
    
    if not (os.path.exists(path_mix) and os.path.exists(path_tgt) and os.path.exists(path_int)):
        print("CRITICAL ERROR: Audio files missing (mixture/target/interference).")
        return

    # 2. Load Audio
    # Load Mix and transpose to (Channels, Samples)
    y_mix, _ = sf.read(path_mix, dtype='float32') 
    if y_mix.ndim > 1 and y_mix.shape[0] > y_mix.shape[1]: 
        y_mix = y_mix.T 
    
    # Load References (Mono)
    s_tgt_ref, _ = sf.read(path_tgt, dtype='float32')
    s_int_ref, _ = sf.read(path_int, dtype='float32')
    
    # 3. Compute STFTs
    _, _, Y_mix = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_tgt = scipy.signal.stft(s_tgt_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_int = scipy.signal.stft(s_int_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP)

    # Ensure Y_mix is (Channels, Freqs, Time)
    if Y_mix.ndim == 2: Y_mix = Y_mix[np.newaxis, :, :] 
    
    # 4. Oracle Mask Generation (Ideal Binary Mask)
    # In reverb, S_int contains the reverb tail. This mask effectively identifies
    # frames where (Interference + Reverb + Noise) > Target.
    mag_tgt = np.abs(S_tgt)
    mag_int = np.abs(S_int)
    mask_noise = np.where(mag_int > mag_tgt, 1.0, 0.0)
    
    print("Oracle Mask generated (Includes Reverb tails in Interference).")

    # 5. Covariance Estimation (Rn)
    # We estimate Rn using only frames where Interference/Noise dominates.
    n_channels, n_freqs, n_frames = Y_mix.shape
    R_noise = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    
    for f_idx in range(n_freqs):
        m_f = mask_noise[f_idx, :] # Shape: (Frames,)
        Y_f = Y_mix[:, f_idx, :]   # Shape: (Channels, Frames)
        
        # Weighted data: effectively zero out frames where Target is dominant
        Y_weighted = Y_f * np.sqrt(m_f) 
        
        # Calculate spatial correlation matrix
        # Normalization: sum(mask) approximates the number of noise-only frames
        normalization = np.sum(m_f) + 1e-6
        R_noise[f_idx] = (Y_weighted @ Y_weighted.conj().T) / normalization

    # 6. MVDR Beamforming Loop
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    
    # Pre-calculate steering vectors to save time inside loop (Optional optimization)
    # But loop is fine for readability.
    
    for f_idx in range(n_freqs):
        freq_hz = f_idx * FS / N_FFT
        
        # Optimization: Skip unstable low frequencies
        if freq_hz < hp_cutoff: 
            continue 

        # Optimization: Diagonal Loading
        # In reverb, the noise field is more diffuse. A higher sigma helps
        # regularize the inversion against the diffuse tail.
        R = R_noise[f_idx] + sigma * np.eye(n_channels)
        
        # Standard Steering Vector (Anechoic approximation)
        d = get_steering_vector(ANGLE_TARGET, freq_hz, D, C)
        
        # MVDR Solution: w = (R^-1 * d) / (d^H * R^-1 * d)
        try:
            R_inv_d = np.linalg.solve(R, d)
            denominator = d.conj().T @ R_inv_d
            w = R_inv_d / (denominator + 1e-10)
        except np.linalg.LinAlgError:
            # Fallback for singularity
            w = np.ones(n_channels) / n_channels

        # Apply weights
        S_mvdr[f_idx, :] = w.conj().T @ Y_mix[:, f_idx, :]

# 7. Post-Filtering: Switch to Ideal Ratio Mask (IRM) for better PESQ
    
    # Calculate Power Spectrograms (Magnitude Squared)
    P_tgt = np.abs(S_tgt)**2
    P_int = np.abs(S_int)**2
    
    # --- Option A: Ideal Ratio Mask (IRM) ---
    # Soft mask: Retains more signal texture, less harsh artifacts
    # mask_soft = sqrt( P_s / (P_s + P_n) )
    mask_soft = np.sqrt(P_tgt / (P_tgt + P_int + 1e-10))
    
    # --- Option B: Wiener Filter (Aggressive but smooth) ---
    # mask_wiener = P_s / (P_s + P_n)
    # mask_soft = P_tgt / (P_tgt + P_int + 1e-10)
    
    # Apply the Soft Mask to the MVDR output
    S_final = S_mvdr * mask_soft


    # 8. Reconstruction
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    # Normalize
    s_out /= (np.max(np.abs(s_out)) + 1e-9)
    
    # 9. Save Output
    # mvdr_out_dir = os.path.join(outdir, "MVDR_Outputs")
    # os.makedirs(mvdr_out_dir, exist_ok=True)
    
    out_filename = "output_oracle_reverb.wav"
    out_path = os.path.join(outdir, out_filename)
    sf.write(out_path, s_out, FS)
    
    print(f"Saved: {out_path}")
    print("-----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optimized Oracle MVDR")
    
    # Directory Argument
    parser.add_argument('--outdir', type=str, default=DEFAULT_OUTDIR, 
                        help='Directory containing the simulation files')
    
    # Optimization Arguments (Experiment Hooks)
    parser.add_argument('--sigma', type=float, default=1e-3, 
                        help='Diagonal Loading Factor (Regularization). Try 1e-3, 1e-2, 1e-1.')
    parser.add_argument('--hp', type=float, default=100.0, 
                        help='High-pass filter cutoff in Hz. Try 50, 100, 200.')
    
    args = parser.parse_args()
    main(args)