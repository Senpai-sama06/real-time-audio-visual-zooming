import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys

# --- 1. Constants ---
FS = 16000
D = 0.01  # Matches World.py (1cm)
C = 343.0
ANGLE_TARGET = 90.0
N_MICS = 2

# MVDR Settings
SIGMA = 1e-7
N_FFT = 512
N_HOP = 256

# --- 2. Helpers ---

def get_steering_vector(angle_deg, f, d, c):
    theta_rad = np.deg2rad(angle_deg)
    phi_rad = 0.0 
    
    tau_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi) / c
    
    omega = 2 * np.pi * f
    d_vec = np.array([
        [np.exp(-1j * omega * tau_m1)],
        [np.exp(-1j * omega * tau_m2)]
    ], dtype=complex)
    return d_vec

def compute_hard_geometric_mask(Y_stft, freqs):
    # Y_stft shape: [Channels, Freq, Time]
    Y1 = Y_stft[0, :, :]
    Y2 = Y_stft[1, :, :]
    
    phase_diff = np.angle(Y1) - np.angle(Y2)
    THRESHOLD = 0.0 
    
    mask_noise = np.where(np.abs(phase_diff) > THRESHOLD, 1.0, 0.01)
    return mask_noise

# --- 3. Main Pipeline ---

def main(output_dir_world):
    # 1. Validation
    if not output_dir_world or not os.path.exists(output_dir_world):
        print(f"ERROR: Invalid directory provided: {output_dir_world}")
        return

    print(f"--- 2. Masked MVDR Processing ---")
    
    # 2. Find Input Audio (Inside World_Outputs)
    input_file = os.path.join(output_dir_world, "mixture_3_sources.wav")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    run_root_dir = os.path.dirname(output_dir_world)
    mvdr_output_dir = os.path.join(run_root_dir, "MVDR_Outputs")
    os.makedirs(mvdr_output_dir, exist_ok=True)

    # --- PROCESSING ---
    
    # Load Audio
    y, fs = sf.read(input_file, dtype='float32')
    y = y.T 
    
    # STFT
    f, t, Y_stft = scipy.signal.stft(y, fs=fs, nperseg=N_FFT, noverlap=N_HOP)
    n_channels, n_freqs, n_frames = Y_stft.shape
    
    # Compute Mask
    print("Calculating Hard Phase Mask...")
    mask_noise = compute_hard_geometric_mask(Y_stft, f)
    
    # Save Mask Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(mask_noise, aspect='auto', origin='lower', cmap='gray')
    plt.title("Hard Noise Mask (White=Noise, Black=Target)")
    mask_plot_path = os.path.join(mvdr_output_dir, "hard_mask.png")
    plt.savefig(mask_plot_path)
    
    # Calculate Covariance
    print("Computing Weighted Noise Covariance...")
    R_noise = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    
    for f_idx in range(n_freqs):
        m_f = mask_noise[f_idx, :] 
        Y_f = Y_stft[:, f_idx, :] 
        
        Y_weighted = Y_f * np.sqrt(m_f) 
        R_curr = Y_weighted @ Y_weighted.conj().T
        
        normalization = np.sum(m_f) + 1e-6 
        R_noise[f_idx] = R_curr / normalization

    # Beamforming
    print(f"Beamforming with SIGMA={SIGMA}...")
    S_out_stft = np.zeros((n_freqs, n_frames), dtype=complex)
    
    for f_idx in range(n_freqs):
        if f[f_idx] < 100: continue 
        
        R = R_noise[f_idx]
        R_loaded = R + SIGMA * np.eye(n_channels)
        d = get_steering_vector(ANGLE_TARGET, f[f_idx], D, C)
        
        try:
            R_inv_d = np.linalg.solve(R_loaded, d)
            denom = d.conj().T @ R_inv_d
            w = R_inv_d / (denom + 1e-10)
        except np.linalg.LinAlgError:
            w = np.array([[1], [0]])
            
        S_out_stft[f_idx, :] = w.conj().T @ Y_stft[:, f_idx, :]
        
    # iSTFT
    _, s_out = scipy.signal.istft(S_out_stft, fs=fs, nperseg=N_FFT, noverlap=N_HOP)
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-6)
    
    # Save Audio Output
    wav_out_path = os.path.join(mvdr_output_dir, "output_masked_mvdr.wav")
    sf.write(wav_out_path, s_out, fs)
    
    print(f"Done.")
    print(f"Saved outputs to: {mvdr_output_dir}")

if __name__ == "__main__":
    main()