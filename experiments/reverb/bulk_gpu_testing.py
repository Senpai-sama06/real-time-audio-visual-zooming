#!/usr/bin/env python3
"""
bulk_testing_gpu.py

Unified bulk comparison script:
1. Generate world (world.py)
2. GPU-based unified covariance + MVDR + SMVB
3. CPU evaluation using eval.py
4. Log results to CSV

This version includes all fixes discussed.
"""

import os
import sys
import csv
import subprocess
import numpy as np
import soundfile as sf
import torch

# ============================================================
# PATH SETUP (CRITICAL)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

WORLD_SCRIPT = os.path.join(SCRIPT_DIR, "world.py")
EVAL_SCRIPT_DIR = SCRIPT_DIR  # eval.py is here

# ============================================================
# IMPORT EVAL FUNCTIONS (CPU)
# ============================================================
from eval import (
    load_and_align_signals,
    calculate_osnr_and_osir,
    calculate_pesq_metric
)

# ============================================================
# CONFIGURATION
# ============================================================
N_SAMPLES = 1
SAVE_DIR = os.path.join(SCRIPT_DIR, "sample")
CSV_PATH = os.path.join(SCRIPT_DIR, "bulk_results.csv")

FS = 16000
N_FFT = 256
N_HOP = 128
D = 0.08
C = 343.0
ANGLE_TARGET = 90.0
SIGMA = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# STFT window (important)
WINDOW = torch.hann_window(N_FFT, device=DEVICE)

# ============================================================
# STEERING VECTOR (GPU, COMPLEX)
# ============================================================
def steering_vector(f_hz: float) -> torch.Tensor:
    theta = torch.tensor(np.deg2rad(ANGLE_TARGET), device=DEVICE)
    omega = 2 * np.pi * f_hz

    tau1 = (D / 2) * torch.cos(theta) / C
    tau2 = (D / 2) * torch.cos(theta - np.pi) / C

    v = torch.stack([
        torch.exp(-1j * omega * tau1),
        torch.exp(-1j * omega * tau2)
    ], dim=0).reshape(2, 1)

    return v.to(torch.complex64)

# ============================================================
# GPU PIPELINE
# ============================================================
def run_beamformers_gpu():
    # ------------------------
    # Load audio (CPU)
    # ------------------------
    y_mix, _ = sf.read(os.path.join(SAVE_DIR, "mixture.wav"), dtype="float32")
    s_t, _ = sf.read(os.path.join(SAVE_DIR, "target.wav"), dtype="float32")
    s_i, _ = sf.read(os.path.join(SAVE_DIR, "interference.wav"), dtype="float32")
    s_n, _ = sf.read(os.path.join(SAVE_DIR, "noise.wav"), dtype="float32")

    # Mono refs
    s_t = s_t[:, 0] if s_t.ndim > 1 else s_t
    s_i = s_i[:, 0] if s_i.ndim > 1 else s_i
    s_n = s_n[:, 0] if s_n.ndim > 1 else s_n

    # To GPU
    Y = torch.tensor(y_mix.T, device=DEVICE)
    s_t = torch.tensor(s_t, device=DEVICE)
    s_i = torch.tensor(s_i, device=DEVICE)
    s_n = torch.tensor(s_n, device=DEVICE)

    # ------------------------
    # STFT (GPU)
    # ------------------------
    Y_stft = torch.stft(Y, N_FFT, N_HOP, window=WINDOW, return_complex=True)
    S_t = torch.stft(s_t, N_FFT, N_HOP, window=WINDOW, return_complex=True)
    S_i = torch.stft(s_i, N_FFT, N_HOP, window=WINDOW, return_complex=True)
    S_n = torch.stft(s_n, N_FFT, N_HOP, window=WINDOW, return_complex=True)

    # ------------------------
    # Oracle masks
    # ------------------------
    mag_t2 = torch.abs(S_t) ** 2
    mag_i2 = torch.abs(S_i) ** 2
    mag_n2 = torch.abs(S_n) ** 2

    mask_t = mag_t2 / (mag_t2 + mag_i2 + mag_n2 + 1e-10)
    mask_in = 1.0 - mask_t

    # ------------------------
    # Covariance estimation
    # ------------------------
    n_freq = Y_stft.shape[1]
    R_in = torch.zeros((n_freq, 2, 2), dtype=torch.complex64, device=DEVICE)

    for f in range(n_freq):
        w = torch.sqrt(mask_in[f])
        Yf = Y_stft[:, f, :]
        Yw = Yf * w
        R_in[f] = (Yw @ Yw.conj().T) / (torch.sum(w**2) + 1e-8)

    # ------------------------
    # Beamforming
    # ------------------------
    S_mvdr = torch.zeros_like(S_t)
    S_smvb = torch.zeros_like(S_t)

    desired = torch.tensor([[1.0], [0.0]],
                            dtype=torch.complex64,
                            device=DEVICE)

    for f in range(n_freq):
        f_hz = f * FS / N_FFT

        if f_hz < 100:
            S_mvdr[f] = Y_stft[0, f]
            S_smvb[f] = Y_stft[0, f]
            continue

        d = steering_vector(f_hz)

        # ----- MVDR -----
        R = R_in[f] + SIGMA * torch.eye(2, device=DEVICE)
        w_mvdr = torch.linalg.solve(R, d)
        w_mvdr /= (d.conj().T @ w_mvdr + 1e-10)
        S_mvdr[f] = (w_mvdr.conj().T @ Y_stft[:, f]).squeeze()

        # ----- SMVB -----
        eigvals, eigvecs = torch.linalg.eigh(R_in[f])
        v_int = eigvecs[:, -1].reshape(2, 1)

        Cmat = torch.cat([d, v_int], dim=1)

        if torch.linalg.cond(Cmat) < 10:
            w_smvb = torch.linalg.solve(Cmat.conj().T, desired)
        else:
            w_smvb = d / 2.0

        S_smvb[f] = (w_smvb.conj().T @ Y_stft[:, f]).squeeze()

    # ------------------------
    # ISTFT + save (CPU)
    # ------------------------
    s_mvdr = torch.istft(S_mvdr * mask_t, N_FFT, N_HOP, window=WINDOW).cpu().numpy()
    s_smvb = torch.istft(S_smvb * mask_t, N_FFT, N_HOP, window=WINDOW).cpu().numpy()

    s_mvdr /= np.max(np.abs(s_mvdr)) + 1e-10
    s_smvb /= np.max(np.abs(s_smvb)) + 1e-10

    sf.write(os.path.join(SAVE_DIR, "output_unified_mvdr.wav"), s_mvdr, FS)
    sf.write(os.path.join(SAVE_DIR, "output_unified_smvb.wav"), s_smvb, FS)

# ============================================================
# MAIN BULK LOOP
# ============================================================
def main():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id",
                         "mvdr_sir", "smvb_sir",
                         "mvdr_pesq", "smvb_pesq"])

        for i in range(N_SAMPLES):
            print(f"\n=== Sample {i} ===")

            # 1. World generation (CPU)
            try:
                subprocess.run([
                    sys.executable, "world.py",
                    "--no-reverb",
                    "--dataset", "ljspeech",
                    "--n", "200"
                ], check=True, capture_output=True, text=True)

            except subprocess.CalledProcessError as e:
                print("--- COMMAND FAILED ---")
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)

            # 2. GPU processing
            run_beamformers_gpu()


# debug required here -|||||
            # 3. Evaluation (MVDR)
            s_est, s_tgt, s_int, _, _ = load_and_align_signals(os.path.join(SAVE_DIR, "output_unified_mvdr.wav"), SAVE_DIR)
            _, mvdr_sir, _, _, _ = calculate_osnr_and_osir(s_est, s_tgt, s_int)
            mvdr_pesq, _ = calculate_pesq_metric(s_tgt, s_est, FS)

            # 4. Evaluation (SMVB)
            s_est, s_tgt, s_int, _, _ = load_and_align_signals(
                os.path.join(SAVE_DIR, "output_unified_smvb.wav"), SAVE_DIR)
            _, smvb_sir, _, _, _ = calculate_osnr_and_osir(s_est, s_tgt, s_int)
            smvb_pesq, _ = calculate_pesq_metric(s_tgt, s_est, FS)

            # 5. Log
            writer.writerow([i,
                             mvdr_sir, smvb_sir,
                             mvdr_pesq, smvb_pesq])

            print(f"[OK] Logged sample {i}")

    print("\n=== BULK EXPERIMENT COMPLETE ===")
    print(f"Results saved to: {CSV_PATH}")

# ============================================================
if __name__ == "__main__":
    main()
