import numpy as np
import scipy.signal
import soundfile as sf
import os
import matplotlib.pyplot as plt
import matplotlib as mlt
mlt.use("TkAgg")

# --- Constants ---
FS = 16000
N_MICS = 2
N_FFT = 256
N_HOP = 128
D = 0.08  
C = 343.0
ANGLE_TARGET = 90.0
SIGMA = 1
SAVE_DIR = "/home/rpzrm/global/projects/real-time-audio-visual-zooming/experiments/reverb/sample"

def get_steering_vector(angle_deg, f, d, c):
    theta_rad = np.deg2rad(angle_deg)
    phi_rad = 0.0
    tau_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi) / c
    omega = 2 * np.pi * f
    d_vec = np.array([[np.exp(-1j * omega * tau_m1)],
                      [np.exp(-1j * omega * tau_m2)]], dtype=complex)
    return d_vec

def load_mono(path):
    """Load audio and force it to mono if stereo."""
    x, _ = sf.read(path)
    if x.ndim > 1:
        x = x[:, 0]   # Take channel 1
    return x.astype(np.float32)

def main():
    print("--- ORACLE TEST: Theoretical Upper Bound Validation ---")

    # Check files
    if not os.path.exists(f"{SAVE_DIR}/target.wav") or not os.path.exists(f"{SAVE_DIR}/interference.wav"):
        print("Error: Reference files missing. Run world.py first.")
        return  

    # Load mixture (stereo)
    y_mix, _ = sf.read(f"{SAVE_DIR}/mixture.wav", dtype='float32')
    y_mix = y_mix.T  # (2, samples)

    # Load clean oracle signals (mono)
    s_tgt_ref = load_mono(f"{SAVE_DIR}/target.wav")
    s_int_ref = load_mono(f"{SAVE_DIR}/interference.wav")

    # --- Compute STFTs (all consistent) ---
    f, t, Y_mix = scipy.signal.stft(
        y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP
    )  # (2, F, T)

    _, _, S_tgt = scipy.signal.stft(
        s_tgt_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP
    )  # (F, T)

    _, _, S_int = scipy.signal.stft(
        s_int_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP
    )  # (F, T)

    # 3. SOFT ORACLE MASK
    mag_tgt = np.abs(S_tgt)
    mag_int = np.abs(S_int)

    eps = 1e-6
    mask_target = mag_tgt / (mag_tgt + mag_int + eps)   # soft mask in [0,1]
    mask_noise = 1.0 - mask_target

    print("Soft mask generated from ground truth magnitude comparison.")

    # 4. Noise covariance estimation
    n_channels, n_freqs, n_frames = Y_mix.shape
    R_noise = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)

    print("Computing noise covariance via soft-masked mixture...")

    for f_idx in range(n_freqs):
        m_f = mask_noise[f_idx, :]  # shape (T,)
        Y_f = Y_mix[:, f_idx, :]    # (2, T)
        Y_weighted = Y_f * np.sqrt(m_f)

        R_noise[f_idx] = (Y_weighted @ Y_weighted.conj().T) / (np.sum(m_f) + 1e-6)

    # 5. MVDR beamformer
    print("Running MVDR...")

    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)

    for f_idx in range(n_freqs):
        # avoid DC/very low frequency
        if f[f_idx] < 100:
            continue

        R = R_noise[f_idx] + SIGMA * np.eye(n_channels)
        d = get_steering_vector(ANGLE_TARGET, f[f_idx], D, C)

        try:
            w = np.linalg.solve(R, d)
            w /= (d.conj().T @ w + 1e-10)
        except:
            w = np.array([[1], [0]])

        S_mvdr[f_idx, :] = w.conj().T @ Y_mix[:, f_idx, :]

    # 6. Apply post-filter (soft mask)
    print("Applying soft post-filter...")
    S_final = S_mvdr * mask_target

    # 7. ISTFT reconstruction
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    s_out /= np.max(np.abs(s_out) + 1e-10)

    sf.write(f"{SAVE_DIR}/output_oracle.wav", s_out, FS)
    print(f"Saved: {SAVE_DIR}/output_oracle.wav")
    print("Run validate.py on this result.")

if __name__ == "__main__":
    main()
