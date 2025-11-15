import numpy as np
import scipy.signal
import scipy.linalg
import soundfile as sf
import os
import matplotlib.pyplot as plt

# --- 1. Constants ---
FS = 16000
D = 0.04
C = 343.0
N_MICS = 2
N_FFT = 512
N_HOP = 256
SCAN_RESOLUTION_DEG = 5 # Scan every 5 degrees. Lower is slower but better.
STABILITY_EPS = 1e-6    # For diagonal loading

# --- 2. "THE KNOB": Field of View (FOV) ---
# This is what you control.
# It will keep everything between (CENTER - WIDTH/2) and (CENTER + WIDTH/2)

FOV_CENTER_DEG = 90.0
FOV_WIDTH_DEG = 3.0  # Try 20 (Zoom In) and 60 (Zoom Out)

# --- 3. Physics Helpers ---

def get_steering_vector(angle_deg, f, d, c):
    """Calculates the steering vector for our 2-mic array."""
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

def calculate_mpdr_power(R_inv, d_vec):
    """Calculates MPDR power: P(theta) = 1 / (d.H * R_inv * d)"""
    denom = d_vec.conj().T @ R_inv_d
    return 1.0 / (denom + 1e-10)

# --- 4. Main Pipeline ---

def main():
    print(f"--- [Test Arch] Max-SNR (GEV) Beamformer ---")
    print(f"FOV Center: {FOV_CENTER_DEG} deg, Width: {FOV_WIDTH_DEG} deg")
    
    fov_min = FOV_CENTER_DEG - (FOV_WIDTH_DEG / 2)
    fov_max = FOV_CENTER_DEG + (FOV_WIDTH_DEG / 2)
    
    input_file = "test_mixture.wav"
    if not os.path.exists(input_file):
        print("Error: test_mixture.wav not found. Run test_world.py first.")
        return

    # 1. Load Audio
    y, fs = sf.read(input_file, dtype='float32')
    y = y.T # [2, N_samples]
    
    # 2. STFT
    f, t, Y_stft = scipy.signal.stft(y, fs=fs, nperseg=N_FFT, noverlap=N_HOP)
    n_channels, n_freqs, n_frames = Y_stft.shape
    
    S_out_stft = np.zeros((n_freqs, n_frames), dtype=complex)
    
    print("Processing frequency bins...")
    for f_idx in range(n_freqs):
        freq = f[f_idx]
        if freq < 100 or freq > 7000: continue # Skip unstable freqs
            
        # --- Stage 1: The "World Scan" ---
        
        # a) Calculate total covariance R for this frequency
        R_total = np.zeros((n_channels, n_channels), dtype=complex)
        for t_idx in range(n_frames):
            Y_fk_t = Y_stft[:, f_idx, t_idx].reshape(n_channels, 1)
            R_total += Y_fk_t @ Y_fk_t.conj().T
        R_total /= n_frames
        
        # Add stability
        R_total += np.eye(n_channels) * STABILITY_EPS
        R_inv = np.linalg.inv(R_total)
        
        # b) Scan all angles to build Rs and Rn
        R_s = np.zeros((n_channels, n_channels), dtype=complex) # Inside FOV
        R_n = np.zeros((n_channels, n_channels), dtype=complex) # Outside FOV
        
        for angle in range(0, 181, SCAN_RESOLUTION_DEG):
            d_vec = get_steering_vector(angle, freq, D, C)
            
            # P(theta) = 1 / (d.H * R_inv * d)
            power_at_angle = 1.0 / (d_vec.conj().T @ R_inv @ d_vec + 1e-10)
            
            # R(theta) = P(theta) * v * v.H
            R_at_angle = power_at_angle[0,0] * (d_vec @ d_vec.conj().T)
            
            # --- Stage 2: The "Integration" (Sort into buckets) ---
            if fov_min <= angle <= fov_max:
                R_s += R_at_angle
            else:
                R_n += R_at_angle

        # --- Stage 3: The "Solution" (Solve GEV) ---
        
        # Add stability to the noise matrix
        R_n += np.eye(n_channels) * STABILITY_EPS
        
        try:
            # Solve GEV: R_s * w = lambda * R_n * w
            eigvals, eigvecs = scipy.linalg.eig(R_s, R_n)
            
            # The filter w is the eigenvector with the largest eigenvalue
            w = eigvecs[:, np.argmax(np.real(eigvals))]
            w = w.reshape(n_channels, 1)
            
        except scipy.linalg.LinAlgError:
            # Fallback: just point at the center (MVDR)
            d = get_steering_vector(FOV_CENTER_DEG, freq, D, C)
            w = np.linalg.solve(R_n, d)
            w /= (d.conj().T @ w + 1e-10)

        # Apply the filter
        S_out_stft[f_idx, :] = w.conj().T @ Y_stft[:, f_idx, :]
        
    # 5. iSTFT
    _, s_out = scipy.signal.istft(S_out_stft, fs=fs, nperseg=N_FFT, noverlap=N_HOP)
    s_out /= (np.max(np.abs(s_out)) + 1e-6)
    
    # Save a unique file
    output_filename = f"output_maxsnr_fov_{FOV_WIDTH_DEG}deg.wav"
    sf.write(output_filename, s_out, fs)
    print(f"\nDone. Output saved to '{output_filename}'")

if __name__ == "__main__":
    main()