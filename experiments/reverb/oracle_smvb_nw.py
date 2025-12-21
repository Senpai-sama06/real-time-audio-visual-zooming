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
MIC_DIST = 0.08
C_SPEED = 343.0
ANGLE_TARGET = 90.0
N_MICS = 2

SAVE_DIR = "/home/rpzrm/global/projects/real-time-audio-visual-zooming/experiments/reverb/sample"


# ========================================================
# Steering Vector
# ========================================================
def get_steering_vector_single(f, angle_deg, d, c):
    theta = np.deg2rad(angle_deg)

    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f

    v = np.array([
        [np.exp(-1j * omega * tau1)],
        [np.exp(-1j * omega * tau2)]
    ], dtype=complex)

    return v / (v[0] + 1e-10)


# ========================================================
# SMVB (Hybrid Null Steering, standalone)
# ========================================================
def hybrid_hard_null_bf(Y, mask, f_bins):
    n_freq = Y.shape[1]
    n_frames = Y.shape[2]

    S_out = np.zeros((n_freq, n_frames), dtype=complex)
    desired = np.array([[1], [0]], dtype=np.complex64)

    mask_int = 1.0 - mask

    for i in range(n_freq):
        f_hz = f_bins[i]

        # Very-low freq fallback
        if f_hz < 200:
            S_out[i, :] = Y[0, i, :]
            continue

        Y_f = Y[:, i, :]         # (2, T)
        m_int = mask_int[i, :]   # (T,)
        

        # made changes such that no masking occurs
        # m_int = 1

        
        denom = np.sum(m_int) + 1e-6
        R_int = (Y_f * m_int) @ (Y_f.conj().T) / denom

        # Dominant interference eigenvector
        eigvals, eigvecs = np.linalg.eigh(R_int)
        v_int = eigvecs[:, -1].reshape(2, 1)
        v_int = v_int / (v_int[0] / (np.abs(v_int[0]) + 1e-10))

        # Target steering
        v_tgt = get_steering_vector_single(f_hz, ANGLE_TARGET, MIC_DIST, C_SPEED)

        # Constraint matrix
        C = np.column_stack((v_tgt, v_int))

        if np.linalg.cond(C) > 10:
            w = v_tgt / N_MICS
        else:
            try:
                w = np.linalg.solve(C.conj().T, desired)
            except:
                w = v_tgt / N_MICS

        S_out[i, :] = (w.conj().T @ Y_f).squeeze()

    return S_out


# ========================================================
# MAIN: Noise-Aware Oracle SMVB
# ========================================================
def main():
    print("=== ORACLE SMVB (Noise-Aware Mask) ===")

    mix_path = f"{SAVE_DIR}/mixture.wav"
    tgt_path = f"{SAVE_DIR}/target.wav"
    int_path = f"{SAVE_DIR}/interference.wav"

    if not os.path.exists(mix_path) or not os.path.exists(tgt_path) or not os.path.exists(int_path):
        print("Missing reference files.")
        return

    # Load mixture
    y_mix, _ = sf.read(mix_path, dtype='float32')
    Y = y_mix.T  # (2, S)

    # Load references (stereo → mono)
    s_t_stereo, _ = sf.read(tgt_path, dtype='float32')
    s_i_stereo, _ = sf.read(int_path, dtype='float32')

    s_t = s_t_stereo[:, 0] if s_t_stereo.ndim > 1 else s_t_stereo
    s_i = s_i_stereo[:, 0] if s_i_stereo.ndim > 1 else s_i_stereo

    # STFT
    f_bins, t_bins, Y_mix = scipy.signal.stft(
        Y, fs=FS, nperseg=N_FFT, noverlap=N_HOP
    )
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

    mask = mag_t**2 / (mag_t**2 + mag_i**2 + mag_n**2 + 1e-8)

    print("Noise-aware oracle mask computed for SMVB.")

    # -----------------------------
    # SMVB
    # -----------------------------
    S_out = hybrid_hard_null_bf(Y_mix, mask, f_bins)

    # Oracle spectral post-filter
    S_final = S_out * mask

    # ISTFT
    _, s_out = scipy.signal.istft(
        S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP
    )
    s_out /= np.max(np.abs(s_out) + 1e-10)

    out_path = f"{SAVE_DIR}/output_oracle_smvb_noiseaware.wav"
    sf.write(out_path, s_out, FS)
    print(f"Saved: {out_path}")

            
if __name__ == "__main__":
    main()
