# masked_mvdr_bully.py
"""
Masked MVDR with Dominant Interference Suppression ("Bully Algorithm")
- Keeps your original pipeline (STFT/iSTFT, soft IPD mask, steering vector)
- Adds an option to reduce the per-bin noise covariance to rank-1 (dominant eigenpair)
  + R_tilde = lambda1 * v1 v1^H + eps * I
  + eps = EPS_FACTOR * lambda1  (numerical invertibility)
- Collision check: collapse only when lambda1/lambda2 is small (< DOM_RATIO_THRESH).
"""

import numpy as np
import scipy.signal
import soundfile as sf
import os

# ---------------------------
# 1. Parameters & Constants
# ---------------------------
FS = 16000
D = 0.08        # mic spacing (meters)
C = 343.0
ANGLE_TARGET = 90.0   # degrees
N_MICS = 2

# STFT settings
N_FFT = 512
N_HOP = 160
WIN = "hann"

# MVDR Settings
ALPHA_COV = 0.97      # covariance smoothing
EPS_LOAD = 1e-6       # base diagonal loading scaling (relative to trace)
# --- Bully algorithm parameters ---
USE_DOMINANT_SUPPRESSION = True
DOM_RATIO_THRESH = 10.0   # if lambda1/lambda2 < this -> consider collision and reduce to rank-1
EPS_FACTOR = 0.01         # eps = EPS_FACTOR * lambda1 (white noise floor when reconstructing rank-1)

# ----------------------------------------------------------
# Steering vector function for linear 2-microphone array
# ----------------------------------------------------------
def steering_vector(angle_deg, freqs, d=D, c=C):
    angle = np.deg2rad(angle_deg)
    tau = (d * np.cos(angle)) / c
    phase = np.exp(-1j * 2 * np.pi * freqs * tau)
    d_vec = np.stack([np.ones_like(phase), phase], axis=-1)  # shape (F,2)
    # normalize (per-frequency) to unit-norm
    norms = np.linalg.norm(d_vec, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    d_vec = d_vec / norms
    return d_vec

# ----------------------------------------------------------
# Load stereo wav into (T,2)
# ----------------------------------------------------------
def load_audio(path):
    x, fs = sf.read(path)
    if x.ndim == 1:
        raise ValueError("Input must be stereo (2-mic signal).")
    if fs != FS:
        raise ValueError(f"Expected {FS} Hz but file has {fs} Hz.")
    return x.astype(np.float32)

# ----------------------------------------------------------
# STFT for both microphones
# Output shape -> (F, T, M)
# ----------------------------------------------------------
def compute_stft(x):
    S_ch = []
    freqs = None
    times = None
    for ch in range(x.shape[1]):
        f, t, Z = scipy.signal.stft(
            x[:, ch], fs=FS, nperseg=N_FFT, noverlap=N_HOP, window=WIN, boundary=None
        )
        S_ch.append(Z)
        freqs = f
        times = t
    # stack to shape (F, T, M)
    S = np.stack(S_ch, axis=-1)  # (F, T, M)
    return S, freqs, times

# ----------------------------------------------------------
# Simple IPD-based target mask (soft)
# ----------------------------------------------------------
def compute_mask(S, angle_target):
    # S: (F,T,2)
    F, T, M = S.shape
    # compute IPD per TF: angle of cross-spectrum
    cross = S[:, :, 0] * np.conj(S[:, :, 1])
    ipd = np.angle(cross)   # shape (F,T)
    # expected IPD at target angle (per frequency)
    freqs = np.linspace(0, FS/2, F)
    steer = steering_vector(angle_target, freqs)[:,1]  # second element's phase
    expected_ipd = np.angle(steer)  # (F,)
    # compute residual (unwrap per freq)
    diff = np.unwrap(ipd - expected_ipd[:, None], axis=1)  # (F,T)
    # soft Gaussian similarity -> mask in [0,1]
    # sigma chosen experimentally; you can tune this
    sigma = 0.8
    mask_target = np.exp(- (diff**2) / (2 * sigma**2))
    mask_target = np.clip(mask_target, 0.0, 1.0)
    return mask_target

# ----------------------------------------------------------
# Dominant interference suppression helper
# ----------------------------------------------------------
def dominant_rank1_reconstruction(R, use_bully=True, ratio_thresh=DOM_RATIO_THRESH, eps_factor=EPS_FACTOR):
    """
    Input:
      R: (M,M) Hermitian covariance matrix
    Output:
      R_tilde: (M,M) modified covariance (either original R or rank-1+eps I)
    Behavior:
      - If use_bully False -> returns R unchanged
      - Otherwise compute EVD: R = V diag(lam) V^H
        If lam1 / lam2 < ratio_thresh -> collision -> return lam1 v1 v1^H + eps*I
        Else return R (already dominated)
    """
    if not use_bully:
        return R

    # ensure Hermitian
    R = 0.5 * (R + R.conj().T)

    # eigen-decomposition (since small M, this is cheap)
    try:
        eigvals, eigvecs = np.linalg.eigh(R)
    except np.linalg.LinAlgError:
        # fallback: small white noise
        M = R.shape[0]
        return R + 1e-12 * np.eye(M)

    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    # pick top two safely (for M=2)
    lambda1 = np.real(eigvals_sorted[0])
    lambda2 = np.real(eigvals_sorted[1]) if eigvals_sorted.shape[0] > 1 else 0.0
    v1 = eigvecs_sorted[:, 0][:, None]  # (M,1)

    # if lambda1 is tiny, don't reconstruct (avoid degenerate)
    if lambda1 <= 1e-12:
        return R

    # collision check: if lambda1 much bigger than lambda2 -> already rank-1 dominated
    # if lambda1/lambda2 < ratio_thresh -> collision => we intervene
    # handle lambda2==0 separately (no collision if lambda2==0)
    do_reconstruct = False
    if lambda2 <= 0:
        # no second component (or negligible)
        do_reconstruct = False
    else:
        ratio = lambda1 / (lambda2 + 1e-20)
        if ratio < ratio_thresh:
            do_reconstruct = True

    if do_reconstruct:
        eps = eps_factor * lambda1
        R_tilde = lambda1 * (v1 @ v1.conj().T) + eps * np.eye(R.shape[0])
        # enforce Hermitian symmetry numeric safety
        R_tilde = 0.5 * (R_tilde + R_tilde.conj().T)
        return R_tilde
    else:
        return R

# ----------------------------------------------------------
# MAIN ADAPTIVE MASKED MVDR with Bully Algorithm
# ----------------------------------------------------------
def masked_mvdr(S, mask_target, d):
    F, T, M = S.shape

    # output STFT
    S_out = np.zeros((F, T), dtype=np.complex128)

    # initialize covariance matrices per freq (tiny identity)
    R = np.tile(np.eye(M)[None,:,:] * 1e-6, (F,1,1))  # (F,M,M)

    for t in range(T):
        y_t = S[:, t, :]  # (F,M)
        m_t = mask_target[:, t]  # (F,)

        # noise confidence (higher -> more noise-only)
        noise_conf = 1.0 - m_t
        noise_conf = np.clip(noise_conf, 1e-3, 1.0)

        for f in range(F):
            y = y_t[f]                          # (M,)
            y_outer = np.outer(y, np.conj(y))   # (M,M)

            # adaptive smoothing (protect when uncertain)
            alpha_eff = ALPHA_COV + (1-ALPHA_COV)*(1-noise_conf[f])

            R[f] = alpha_eff*R[f] + (1-alpha_eff)*noise_conf[f]*y_outer

            # enforce Hermitian
            R[f] = 0.5 * (R[f] + R[f].conj().T)

            # adaptive diagonal loading relative to trace
            tr = np.real(np.trace(R[f]))
            load = EPS_LOAD * tr / M + 1e-12
            R_loaded = R[f] + load * np.eye(M)

            # ---- Bully algorithm: dominant suppression ----
            R_tilde = dominant_rank1_reconstruction(
                R_loaded, use_bully=USE_DOMINANT_SUPPRESSION,
                ratio_thresh=DOM_RATIO_THRESH, eps_factor=EPS_FACTOR
            )

            # robust inversion
            try:
                R_inv = np.linalg.inv(R_tilde)
            except np.linalg.LinAlgError:
                R_inv = np.linalg.pinv(R_tilde)

            df = d[f][:, None]     # (M,1)
            denom = np.conj(df).T @ R_inv @ df
            denom = denom[0,0]
            if np.abs(denom) < 1e-12:
                # fallback: delay-and-sum-ish (preserve target)
                w = df[:, 0].conj()
            else:
                w_vec = (R_inv @ df)[:, 0] / denom  # (M,)
                w = np.conj(w_vec)

            S_out[f,t] = w @ y

    return S_out

# ----------------------------------------------------------
# SAVE OUTPUT
# ----------------------------------------------------------
def istft_and_save(S_out, filename):
    _, x = scipy.signal.istft(
        S_out, fs=FS, nperseg=N_FFT, noverlap=N_HOP, window=WIN, boundary=None
    )
    x = x / (np.max(np.abs(x)) + 1e-8)
    sf.write(filename, x, FS)
    print(f"Saved: {filename}")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    infile = "samples/mixture_3_sources.wav"
    assert os.path.exists(infile), "Missing samples/input_mix.wav"

    x = load_audio(infile)
    S, freqs, _ = compute_stft(x)  # (F,T,M)

    # steering vector for target
    d = steering_vector(ANGLE_TARGET, freqs)

    # mask
    mask_target = compute_mask(S, ANGLE_TARGET)

    # MVDR (with bully)
    S_out = masked_mvdr(S, mask_target, d)

    # save result
    istft_and_save(S_out, "samples/output_adaptive_mvdr_bully.wav")

    print("✅ Adaptive Masked MVDR (with Dominant Interference Suppression) Completed")

if __name__ == "__main__":
    main()
