import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
import os

# --- 1. Constants ---
FS = 16000
D = 0.01  # Changed from 0.04 to 0.01
C = 343.0
ANGLE_TARGET = 90.0
N_MICS = 2

# MVDR Settings
SIGMA = 1e-7  # Lower sigma = Deeper Nulls (Sharper Zoom)
N_FFT = 512
N_HOP = 256

# --- 2. Helpers ---

def get_steering_vector(angle_deg, f, d, c):
    """Calculates the steering vector for the target angle."""
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
    """
    Computes a BINARY mask based on Phase Difference.
    If phase diff > threshold -> NOISE (1.0)
    Else -> TARGET (0.0)
    """
    # Y_stft shape: [Channels, Freq, Time]
    Y1 = Y_stft[0, :, :]
    Y2 = Y_stft[1, :, :]
    
    # 1. Calculate Phase Difference
    phase_diff = np.angle(Y1) - np.angle(Y2)
    
    # 2. The Threshold (in Radians)
    # 0.3 radians is approx 17 degrees. 
    # Anything outside +/- 17 degrees phase error is marked as NOISE.
    THRESHOLD = 0.0 
    
    # Create Binary Mask (1 = Noise, 0 = Target)
    mask_noise = np.where(np.abs(phase_diff) > THRESHOLD, 1.0, 0.01)
    
    # 3. Frequency Cutoff (High freqs are unreliable)
    # for i, f in enumerate(freqs):
    #     if f > 3500: 
    #         mask_noise[i, :] = 1.0 # Treat all high freqs as noise candidates
            
    return mask_noise

# --- 3. Main Pipeline ---

def main():
    print("--- 2. Masked MVDR (Aggressive Hard Mask) ---")
    
    input_file = "mixture_3_sources.wav"
    if not os.path.exists(input_file):
        print("Error: mixture_3_sources.wav not found.")
        return

    # 1. Load Audio
    y, fs = sf.read(input_file, dtype='float32')
    y = y.T 
    
    # 2. STFT
    f, t, Y_stft = scipy.signal.stft(y, fs=fs, nperseg=N_FFT, noverlap=N_HOP)
    n_channels, n_freqs, n_frames = Y_stft.shape
    
    # 3. Compute the Aggressive Mask
    print("Calculating Hard Phase Mask...")
    mask_noise = compute_hard_geometric_mask(Y_stft, f)
    
    # Optional: Visualize
    plt.figure(figsize=(10, 4))
    plt.imshow(mask_noise, aspect='auto', origin='lower', cmap='gray')
    plt.title("Hard Noise Mask (White=Noise, Black=Target)")
    plt.savefig("hard_mask.png")
    print("Saved 'hard_mask.png'")

    # 4. Calculate Masked Covariance Matrix
    print("Computing Weighted Noise Covariance...")
    R_noise = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    
    for f_idx in range(n_freqs):
        m_f = mask_noise[f_idx, :] 
        Y_f = Y_stft[:, f_idx, :] 
        
        # Weighted Covariance Calculation
        # We square the mask to emphasize the binary nature
        Y_weighted = Y_f * np.sqrt(m_f) 
        R_curr = Y_weighted @ Y_weighted.conj().T
        
        normalization = np.sum(m_f) + 1e-6 
        R_noise[f_idx] = R_curr / normalization

    # 5. Beamforming Loop
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
        
    # 6. iSTFT
    _, s_out = scipy.signal.istft(S_out_stft, fs=fs, nperseg=N_FFT, noverlap=N_HOP)
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-6)
    
    sf.write("output_masked_mvdr.wav", s_out, fs)
    print("Done. Saved 'output_masked_mvdr.wav'.")

if __name__ == "__main__":
    main()