#!/usr/bin/env python3
"""
blind_statistical_mvdr.py
A blind approach that estimates covariance from the mixture only.
No ground-truth masks or external noise files are used.
"""

import os
import numpy as np
import soundfile as sf
import scipy.signal

# ------------------------
# Config / Constants
# ------------------------
FS = 16000
N_FFT = 256
N_HOP = 128
D = 0.08            # mic spacing
C = 343.0
ANGLE_TARGET = 90.0
SIGMA = 1e-3        # diagonal loading for stability
SAVE_DIR = "sample"

# ------------------------
# Steering vector
# ------------------------
def get_steering_vector(angle_deg, f, d, c):
    theta = np.deg2rad(angle_deg)
    # Using the same geometry as your reference code
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f
    v = np.array([[np.exp(-1j * omega * tau1)], [np.exp(-1j * omega * tau2)]], dtype=complex)
    return v

# ------------------------
# Main
# ------------------------
def main():
    mix_path = os.path.join(SAVE_DIR, "mixture.wav")

    if not os.path.exists(mix_path):
        print(f"Missing file: {mix_path}")
        return

    # Load stereo mixture (shape: samples, 2)
    y_mix, sr = sf.read(mix_path, dtype="float32")
    if sr != FS: print(f"Warning: mixture SR {sr} != FS {FS}")
    
    if y_mix.ndim == 1:
        print("Error: mixture.wav is mono. Blind MVDR requires 2 channels.")
        return
    
    # Shape for STFT: (2, samples)
    Y_input = y_mix.T  

    # STFT to get frequency domain representation
    f_bins, t_bins, Y_mix = scipy.signal.stft(Y_input, fs=FS, nperseg=N_FFT, noverlap=N_HOP)

    n_channels, n_freqs, n_frames = Y_mix.shape

    # 1. Estimate Total Data Covariance (Ryy) blindly from mixture
    # Ryy(f) = E[y(f,t) y^H(f,t)]
    Ryy = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for f_idx in range(n_freqs):
        Yf = Y_mix[:, f_idx, :]  # (2, T)
        # Standard Sample Matrix Inversion (SMI) approach
        # Equivalent to Equation 15 in the PoC
        Ryy[f_idx] = (Yf @ Yf.conj().T) / n_frames

    # 2. MVDR beamforming
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    
    for f_idx in range(n_freqs):
        f_hz = f_bins[f_idx]
        
        # Lower frequency bypass to avoid noise amplification
        if f_hz < 100:
            S_mvdr[f_idx, :] = Y_mix[0, f_idx, :]
            continue

        # Add Diagonal Loading to the estimated covariance
        # This makes the MVDR "Robust" (RMVB style)
        R = Ryy[f_idx] + SIGMA * np.eye(n_channels)
        
        # Nominal target steering vector
        d = get_steering_vector(ANGLE_TARGET, f_hz, D, C)
        
        try:
            # Solve for w: (R^-1 * d) / (d^H * R^-1 * d)
            # We use linalg.solve for numerical efficiency/stability
            w = np.linalg.solve(R, d)
            w = w / (d.conj().T @ w + 1e-12)
        except np.linalg.LinAlgError:
            # Fallback to mic-0 if matrix is singular
            w = np.array([[1.0], [0.0]], dtype=complex)

        # Apply spatial filter: z = w^H * Y
        S_mvdr[f_idx, :] = (w.conj().T @ Y_mix[:, f_idx, :]).squeeze()

    # 3. ISTFT back to time domain
    _, s_out = scipy.signal.istft(S_mvdr, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    # Normalize output
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-10)

    out_path = os.path.join(SAVE_DIR, "output_blind_mvdr.wav")
    sf.write(out_path, s_out.astype(np.float32), FS)
    print(f"Saved Blind MVDR result: {out_path}")

if __name__ == "__main__":
    main()