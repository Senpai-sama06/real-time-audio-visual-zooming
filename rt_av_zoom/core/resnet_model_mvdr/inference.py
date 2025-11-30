import numpy as np
import torch
import soundfile as sf
import scipy.signal
import json
import os
import time
import torch.nn as nn
import torch.nn.functional as F
from mir_eval.separation import bss_eval_sources

# =====================================================================
# 1. LOAD CONFIG
# =====================================================================
if not os.path.exists("config.json"):
    raise FileNotFoundError("Config missing")

with open("config.json", "r") as f:
    CONF = json.load(f)

FS = CONF["fs"]
N_FFT = CONF["n_fft"]
HOP = CONF["hop_len"]
WIN_SIZE_SAMPLES = CONF["train_seg_samples"]
D = CONF["d"]
C = CONF["c"]
ANGLE_TARGET = 90.0
SIGMA = 1e-5
N_MICS = 2

# Detect device
device = torch.device("cpu")
print(f"Inference running on: {device}")

# =====================================================================
# 2. MODEL ARCHITECTURE (DeepFPU)
# =====================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class DeepFPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))

        self.enc1_conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.enc2_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64)
        )

        self.enc3_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128)
        )

        self.enc4_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            ResBlock(512),
            ResBlock(512)
        )

        self.up4 = nn.ConvTranspose2d(512, 256, (1, 2), stride=(1, 2))
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2))
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def _match(self, x, target):
        if x.shape[3] != target.shape[3]:
            x = F.interpolate(x, size=target.shape[2:], mode='nearest')
        return x

    def forward(self, x):
        e1 = self.enc1_conv(x)
        e2 = self.enc2_conv(self.pool(e1))
        e3 = self.enc3_conv(self.pool(e2))
        e4 = self.enc4_conv(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        u4 = self._match(self.up4(b), e4)
        d4 = self.dec4_conv(torch.cat([u4, e4], dim=1))

        u3 = self._match(self.up3(d4), e3)
        d3 = self.dec3_conv(torch.cat([u3, e3], dim=1))

        u2 = self._match(self.up2(d3), e2)
        d2 = self.dec2_conv(torch.cat([u2, e2], dim=1))

        u1 = self._match(self.up1(d2), e1)
        d1 = self.dec1_conv(torch.cat([u1, e1], dim=1))

        return self.out(d1).squeeze(1)

# =====================================================================
# 3. PHYSICS + UTILITIES
# =====================================================================
def get_steering_vector(angle_deg, f, d, c):
    theta = np.deg2rad(angle_deg)
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f
    return np.array([[np.exp(-1j * omega * tau1)], [np.exp(-1j * omega * tau2)]])

# =====================================================================
# 4. PROCESS A 2-SECOND WINDOW (with inference timing)
# =====================================================================
def process_chunk(y_chunk, model, chunk_idx):
    """Runs MVDR + Mask Estimation on one chunk and prints timing"""
    print(f"\n--- Processing chunk {chunk_idx} ---")

    # STFT
    f, t, Y = scipy.signal.stft(y_chunk.T, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP)

    mag = np.abs(Y)
    ipd = np.angle(Y[0]) - np.angle(Y[1])

    X = torch.from_numpy(np.stack([
        np.log(mag[0] + 1e-7),
        ipd
    ], axis=0)).float().unsqueeze(0).to(device)

    # ------------------ MODEL INFERENCE TIME -----------------------
    tic = time.time()
    with torch.no_grad():
        Mask = model(X).squeeze(0)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_infer = time.time() - tic
    print(f"Mask Estimation Time: {t_infer*1000:.2f} ms")

    Mask = Mask.cpu().numpy()
    Mask_N = 1 - Mask

    # ------------------ MVDR PROCESSING TIME -----------------------
    tic = time.time()

    S_out = np.zeros_like(Y[0], dtype=complex)

    for i in range(Y.shape[1]):
        if f[i] < 100:
            continue

        m_vec = Mask_N[i, :]
        Y_vec = Y[:, i, :]

        R_n = (Y_vec * np.sqrt(m_vec)) @ (Y_vec * np.sqrt(m_vec)).conj().T
        R_n = R_n / (np.sum(m_vec) + 1e-6) + SIGMA * np.eye(N_MICS)

        d = get_steering_vector(ANGLE_TARGET, f[i], D, C)

        try:
            w = np.linalg.solve(R_n, d)
            w /= (d.conj().T @ w + 1e-10)
        except:
            w = np.array([[1], [0]])

        S_out[i, :] = w.conj().T @ Y_vec

    t_mvdr = time.time() - tic
    print(f"MVDR Processing Time: {t_mvdr*1000:.2f} ms")

    # ISTFT
    S_final = S_out * np.maximum(Mask, 0.05)
    _, out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP)

    return out, t_infer, t_mvdr

# =====================================================================
# 5. MAIN DEPLOY
# =====================================================================
def main_deploy(input_path, FOLDER):

    print(f"\nProcessing {input_path} ...")

    if not os.path.exists("mask_estimator.pth"):
        print("Model not found.")
        return

    # Load audio
    y_full, fs = sf.read(input_path, dtype='float32')
    WIN_SIZE = WIN_SIZE_SAMPLES
    HOP = WIN_SIZE // 2

    # Load model
    model = DeepFPU().to(device)
    model.load_state_dict(torch.load("mask_estimator.pth", map_location=device))
    model.eval()

    output_buffer = np.zeros(len(y_full) + WIN_SIZE)
    norm_buffer = np.zeros(len(y_full) + WIN_SIZE)

    num_chunks = int(np.ceil(len(y_full) / HOP))
    print(f"Audio Length: {len(y_full)/FS:.2f}s | Chunks: {num_chunks}")

    total_infer_time = 0
    total_mvdr_time = 0
    total_start = time.time()

    for i in range(num_chunks):
        start = i * HOP
        end = start + WIN_SIZE

        chunk = y_full[start:end]
        if len(chunk) < WIN_SIZE:
            chunk = np.pad(chunk, ((0, WIN_SIZE - len(chunk)), (0, 0)))

        out_chunk, t_inf, t_mvdr = process_chunk(chunk, model, i)

        total_infer_time += t_inf
        total_mvdr_time += t_mvdr

        L = min(len(out_chunk), WIN_SIZE)
        output_buffer[start:start+L] += out_chunk[:L]
        norm_buffer[start:start+L] += 1.0

    total_time = time.time() - total_start

    # Final output
    norm_buffer[norm_buffer == 0] = 1.0
    final_output = output_buffer[:len(y_full)] / norm_buffer[:len(y_full)]

    out_name = f"{FOLDER}/enhanced_{os.path.basename(input_path)}"
    sf.write(out_name, final_output, fs)

    print("\n================ Timing Summary ================")
    print(f"Total Mask Estimation Time: {total_infer_time:.3f} s")
    print(f"Total MVDR Time:            {total_mvdr_time:.3f} s")
    print(f"Total End-to-End Time:      {total_time:.3f} s")
    print("================================================")
    print(f"Saved output: {out_name}")

# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    HOME_DIR = "/home/rpzrm/global/projects/real-time-audio-visual-zooming/rt_av_zoom/core/simulation_results/ljspeech_reverb_20251130_155519/"
    INPUT_FILE = f"{HOME_DIR}/mixture.wav"
    if os.path.exists(INPUT_FILE):
        main_deploy(INPUT_FILE, HOME_DIR)
    else:
        print("Input file not found.")
