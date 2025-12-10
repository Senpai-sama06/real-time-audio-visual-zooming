import numpy as np
import scipy.signal
import soundfile as sf
import os

# ------------------------
# Constants
# ------------------------
FS = 16000
N_FFT = 256
N_HOP = 128
D = 0.08
C = 343.0
ANGLE_TARGET = 90.0
SIGMA = 1e-3   # diagonal loading

SAVE_DIR = "/home/rpzrm/global/projects/real-time-audio-visual-zooming/experiments/reverb/sample"


# ---------------------------------------------------------
# Steering vector (same physics as oracle_debug/inference)
# ---------------------------------------------------------
def get_steering_vector(angle_deg, f, d, c):
    theta = np.deg2rad(angle_deg)
    tau_m1 = (d / 2) * np.cos(theta) / c
    tau_m2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f

    v = np.array([
        [np.exp(-1j * omega * tau_m1)],
        [np.exp(-1j * omega * tau_m2)]
    ])
    return v


# ---------------------------------------------------------
# MAIN: Noise-Aware Oracle MVDR
# ---------------------------------------------------------
def main():
    print("=== ORACLE MVDR (Noise-Aware Mask) ===")

    # File checks
    mix_path = f"{SAVE_DIR}/mixture.wav"
    tgt_path = f"{SAVE_DIR}/target.wav"
    int_path = f"{SAVE_DIR}/interference.wav"

    if not os.path.exists(mix_path):
        print("mixture.wav missing.")
        return
    if not os.path.exists(tgt_path):
        print("target.wav missing.")
        return
    if not os.path.exists(int_path):
        print("interference.wav missing.")
        return

    # Load mixture (stereo)
    y_mix, _ = sf.read(mix_path, dtype='float32')
    Y = y_mix.T  # shape (2, S)

    # Load target/interference references and convert stereo → mono
    s_t_stereo, _ = sf.read(tgt_path, dtype='float32')
    s_i_stereo, _ = sf.read(int_path, dtype='float32')

    s_t = s_t_stereo[:, 0] if s_t_stereo.ndim > 1 else s_t_stereo
    s_i = s_i_stereo[:, 0] if s_i_stereo.ndim > 1 else s_i_stereo

    # --- STFT ---
    f_bins, t_bins, Y_mix = scipy.signal.stft(
        Y, fs=FS, nperseg=N_FFT, noverlap=N_HOP
    )  # (2,F,T)

    _, _, S_t = scipy.signal.stft(s_t, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_i = scipy.signal.stft(s_i, fs=FS, nperseg=N_FFT, noverlap=N_HOP)

    # -----------------------------
    # TRUE NOISE-AWARE ORACLE MASK
    # -----------------------------
    mag_mix = np.abs(Y_mix[0])
    mag_t = np.abs(S_t)
    mag_i = np.abs(S_i)

    mag_n = mag_mix - (mag_t + mag_i)
    mag_n = np.clip(mag_n, 0.0, None)

    mask_t = mag_t**2 / (mag_t**2 + mag_i**2 + mag_n**2 + 1e-8)
    mask_n = 1.0 - mask_t

    print("Noise-aware oracle mask computed for MVDR.")

    # -----------------------------
    # Estimate noise+interference covariance
    # -----------------------------
    n_channels, n_freq, n_frames = Y_mix.shape
    Rn = np.zeros((n_freq, n_channels, n_channels), dtype=complex)

    for f_idx in range(n_freq):
        w = np.sqrt(mask_n[f_idx])  # (T,)
        Y_f = Y_mix[:, f_idx, :]    # (2, T)
        Y_w = Y_f * w

        Rn[f_idx] = (Y_w @ Y_w.conj().T) / (np.sum(mask_n[f_idx]) + 1e-6)

    # -----------------------------
    # MVDR Beamforming
    # -----------------------------
    S_mvdr = np.zeros((n_freq, n_frames), dtype=complex)

    for f_idx in range(n_freq):
        f_val = f_bins[f_idx]
        if f_val < 100:
            S_mvdr[f_idx, :] = Y_mix[0, f_idx, :]
            continue

        R = Rn[f_idx] + SIGMA * np.eye(n_channels)
        d = get_steering_vector(ANGLE_TARGET, f_val, D, C)

        try:
            w = np.linalg.solve(R, d)
            w /= (d.conj().T @ w + 1e-10)
        except:
            w = np.array([[1], [0]])

        S_mvdr[f_idx, :] = (w.conj().T @ Y_mix[:, f_idx, :])

    # Oracle spectral post-filter
    S_final = S_mvdr * mask_t

    # ISTFT
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    s_out /= np.max(np.abs(s_out) + 1e-10)

    out_path = f"{SAVE_DIR}/output_oracle_mvdr_noiseaware.wav"
    sf.write(out_path, s_out, FS)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
