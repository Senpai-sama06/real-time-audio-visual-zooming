#!/usr/bin/env python3
"""
oracle_mvdr_noiseaware_with_noisewav.py
Noise-aware Oracle MVDR that uses explicit noise.wav for mask construction.
Saves: sample/output_oracle_mvdr_noiseaware.wav
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
SIGMA = 1e-3        # diagonal loading for MVDR
SAVE_DIR = "sample"

# ------------------------
# Steering vector
# ------------------------
def get_steering_vector(angle_deg, f, d, c):
    theta = np.deg2rad(angle_deg)
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
    tgt_path = os.path.join(SAVE_DIR, "target.wav")
    int_path = os.path.join(SAVE_DIR, "interference.wav")
    noise_path = os.path.join(SAVE_DIR, "noise.wav")

    for p in (mix_path, tgt_path, int_path, noise_path):
        if not os.path.exists(p):
            print(f"Missing file: {p}. Run world.py first.")
            return

    # Load stereo mixture (shape: (samples, 2))
    y_mix, sr = sf.read(mix_path, dtype="float32")
    if sr != FS: print(f"Warning: mixture SR {sr} != FS {FS}")
    # Convert to (2, samples) as expected by STFT call below
    if y_mix.ndim == 1:
        print("Error: mixture.wav is mono. Expecting stereo 2-channel.")
        return
    Y_input = y_mix.T  # (2, samples)

    # Load target / interference / noise as stereo -> mono (take mic-0)
    s_t_st, _ = sf.read(tgt_path, dtype="float32")
    s_i_st, _ = sf.read(int_path, dtype="float32")
    s_n_st, _ = sf.read(noise_path, dtype="float32")

    s_t = s_t_st[:, 0] if s_t_st.ndim > 1 else s_t_st
    s_i = s_i_st[:, 0] if s_i_st.ndim > 1 else s_i_st
    s_n = s_n_st[:, 0] if s_n_st.ndim > 1 else s_n_st

    # STFTs
    f_bins, t_bins, Y_mix = scipy.signal.stft(Y_input, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_t = scipy.signal.stft(s_t, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_i = scipy.signal.stft(s_i, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_n = scipy.signal.stft(s_n, fs=FS, nperseg=N_FFT, noverlap=N_HOP)

    # Magnitudes
    mag_t = np.abs(S_t)
    mag_i = np.abs(S_i)
    mag_n = np.abs(S_n)

    # True noise-aware Wiener-style oracle mask (power domain)
    eps = 1e-10
    mask_t = (mag_t ** 2) / (mag_t ** 2 + mag_i ** 2 + mag_n ** 2 + eps)
    mask_not_t = 1.0 - mask_t  # used to weight interference+noise covariance

    n_channels, n_freqs, n_frames = Y_mix.shape

    # Estimate noise+interference covariance using mask_not_t
    Rn = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for f_idx in range(n_freqs):
        w = np.sqrt(mask_not_t[f_idx, :])  # shape (T,)
        Yf = Y_mix[:, f_idx, :]            # (2, T)
        Yw = Yf * w                        # broadcast (2, T)
        denom = np.sum(w ** 2) + 1e-8
        # print(np.shape(denom))
        # denom = 1
        Rn[f_idx] = (Yw @ Yw.conj().T) / denom

    # MVDR beamforming
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    for f_idx in range(n_freqs):
        f_hz = f_bins[f_idx]
        # low-freq bypass
        if f_hz < 100:
            S_mvdr[f_idx, :] = Y_mix[0, f_idx, :]
            continue

        R = Rn[f_idx] + SIGMA * np.eye(n_channels)
        d = get_steering_vector(ANGLE_TARGET, f_hz, D, C)
        try:
            w = np.linalg.solve(R, d)
            w = w / (d.conj().T @ w + 1e-10)
        except np.linalg.LinAlgError:
            w = np.array([[1.0], [0.0]], dtype=complex)

        S_mvdr[f_idx, :] = (w.conj().T @ Y_mix[:, f_idx, :]).squeeze()

    # Apply oracle spectral post-filter (mask_t)
    S_final = S_mvdr * mask_t

    # ISTFT
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    # Normalize
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-10)

    out_path = os.path.join(SAVE_DIR, "output_oracle_nw_mvdr.wav")
    sf.write(out_path, s_out.astype(np.float32), FS)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
