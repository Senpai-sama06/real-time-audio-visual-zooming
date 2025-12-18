#!/usr/bin/env python3
"""
oracle_mvdr_hrnr_gated.py
Implements Oracle MVDR Beamforming + GATED Plapous Harmonic Regeneration.

Major Change: "Patch 2" (The Bouncer)
- Calculates Frame-level SNR using the Oracle Noise Projection.
- If SNR < 5dB, the Harmonic Regenerator is DISABLED (Ghost = 0).
- This prevents the Rectifier from regenerating harmonics for the Interference.

Saves: sample/output_oracle_mvdr_hrnr_gated.wav
"""

import os
import numpy as np
import soundfile as sf
import scipy.signal

# ------------------------
# Config / Constants
# ------------------------
FS = 16000
N_FFT = 512
N_HOP = 256
D = 0.08
C = 343.0
ANGLE_TARGET = 90.0
SIGMA = 1e-3   
SAVE_DIR = "sample"

# HRNR Parameters
RHO = 0.95        # Patch 3 integrated: Trust real signal 95%, Ghost 5% (Conservative)
SNR_GATE_DB = 5.0 # Patch 2: Only regenerate if frame SNR > 5dB

# ------------------------
# HRNR (Plapous) Logic with Gating
# ------------------------
class HarmonicRestorerGated:
    def __init__(self, fs, n_fft, n_hop, rho=0.95, snr_gate_db=20.0):
        self.fs = fs
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.rho = rho
        self.snr_gate_db = snr_gate_db

    def restore(self, S_beamformed, gamma_nn):
        """
        S_beamformed: (n_freq, n_frames) Complex STFT
        gamma_nn:     (n_freq,) or (n_freq, n_frames) Projected Noise Variance
        """
        print(f"Running GATED Harmonic Regeneration (Gate > {self.snr_gate_db}dB)...")
        
        # 1. Calculate Frame-Level SNR for Gating
        # P_signal_frame: (n_frames,)
        P_signal_frame = np.mean(np.abs(S_beamformed)**2, axis=0)
        
        # Handle gamma shape (broadcast if static)
        if gamma_nn.ndim == 1:
            P_noise_frame = np.mean(gamma_nn) # Scalar
        else:
            P_noise_frame = np.mean(gamma_nn, axis=0) # (n_frames,)
            
        # Avoid div/0
        frame_snr_linear = P_signal_frame / (P_noise_frame + 1e-10)
        frame_snr_db = 10 * np.log10(frame_snr_linear + 1e-10)
        
        # 2. Create the "Bouncer" Mask (1.0 = Open, 0.0 = Closed)
        # We smooth it slightly to prevent clicks
        gate_mask = (frame_snr_db > self.snr_gate_db).astype(float)
        # Optional: Median filter to remove rapid flickering of the gate
        gate_mask = scipy.signal.medfilt(gate_mask, kernel_size=5)
        
        print(f"  - Frames Regenerated: {np.sum(gate_mask)} / {len(gate_mask)}")

        # 3. ISTFT to get Time Domain
        _, s_time = scipy.signal.istft(S_beamformed, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_hop)
        
        # 4. Rectification (The Ghost Generator)
        s_harmo_time = np.maximum(s_time, 0)
        
        # 5. STFT back to Frequency Domain
        _, _, S_harmo = scipy.signal.stft(s_harmo_time, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_hop)
        
        # Align shapes
        min_frames = min(S_beamformed.shape[1], S_harmo.shape[1])
        S_beamformed = S_beamformed[:, :min_frames]
        S_harmo = S_harmo[:, :min_frames]
        gate_mask = gate_mask[:min_frames]
        
        # Apply Gate to the Ghost Signal
        # If gate is 0, S_harmo becomes 0 (Disabling the Trojan Horse)
        S_harmo = S_harmo * gate_mask[np.newaxis, :]
        
        if gamma_nn.ndim == 1:
            gamma_nn = gamma_nn[:, np.newaxis]
        gamma_nn = gamma_nn[:, :min_frames]

        # 6. Calculate Powers
        P_real = np.abs(S_beamformed)**2
        P_ghost = np.abs(S_harmo)**2
        
        # 7. Calculate Harmonic SNR
        # Note: If gate is closed, P_ghost is 0, so SNR relies purely on Real signal.
        # This falls back to standard Wiener filtering.
        numerator = (self.rho * P_real) + ((1 - self.rho) * P_ghost)
        
        # Patch 1 (Mini): Safety Factor on Noise (2x) just to be safe
        denominator = (gamma_nn * 2.0) + 1e-10
        
        SNR_harmo = numerator / denominator
        
        # 8. Final Gain
        G_harmo = SNR_harmo / (1 + SNR_harmo)
        S_out = S_beamformed * G_harmo
        
        return S_out

# ------------------------
# Utilities
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

    # Load Audio
    y_mix, sr = sf.read(mix_path, dtype="float32")
    if sr != FS: print(f"Warning: mixture SR {sr} != FS {FS}")
    if y_mix.ndim == 1:
        print("Error: mixture.wav is mono. Expecting stereo 2-channel.")
        return
    Y_input = y_mix.T 

    # Load References for Oracle Covariance
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

    mag_t = np.abs(S_t)
    mag_i = np.abs(S_i)
    mag_n = np.abs(S_n)

    # Oracle Mask Construction
    eps = 1e-10
    mask_t = (mag_t ** 2) / (mag_t ** 2 + mag_i ** 2 + mag_n ** 2 + eps)
    mask_not_t = 1.0 - mask_t 

    n_channels, n_freqs, n_frames = Y_mix.shape

    # 1. Estimate Covariance (Rn)
    Rn = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for f_idx in range(n_freqs):
        w = np.sqrt(mask_not_t[f_idx, :])
        Yf = Y_mix[:, f_idx, :]            
        Yw = Yf * w                        
        denom = np.sum(w ** 2) + 1e-8
        Rn[f_idx] = (Yw @ Yw.conj().T) / denom

    # 2. MVDR Beamforming & Noise Projection
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    gamma_nn_projected = np.zeros((n_freqs,), dtype=float)

    for f_idx in range(n_freqs):
        f_hz = f_bins[f_idx]
        if f_hz < 100:
            S_mvdr[f_idx, :] = Y_mix[0, f_idx, :]
            gamma_nn_projected[f_idx] = 1.0 
            continue

        R = Rn[f_idx] + SIGMA * np.eye(n_channels)
        d = get_steering_vector(ANGLE_TARGET, f_hz, D, C)
        try:
            w_mvdr = np.linalg.solve(R, d)
            w_mvdr = w_mvdr / (d.conj().T @ w_mvdr + 1e-10)
        except np.linalg.LinAlgError:
            w_mvdr = np.array([[1.0], [0.0]], dtype=complex)

        S_mvdr[f_idx, :] = (w_mvdr.conj().T @ Y_mix[:, f_idx, :]).squeeze()
        
        # Projection
        noise_leakage = (w_mvdr.conj().T @ Rn[f_idx] @ w_mvdr).real
        gamma_nn_projected[f_idx] = float(noise_leakage)

    # 3. GATED Harmonic Restoration
    # Note: increased rho to 0.95 (conservative) and Gate to 5dB
    hrnr = HarmonicRestorerGated(FS, N_FFT, N_HOP, rho=0.95, snr_gate_db=5.0)
    S_final = hrnr.restore(S_mvdr, gamma_nn_projected)

    # ISTFT & Save
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-10)

    out_path = os.path.join(SAVE_DIR, "output_oracle_mvdr_hrnr_gated.wav")
    sf.write(out_path, s_out.astype(np.float32), FS)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()