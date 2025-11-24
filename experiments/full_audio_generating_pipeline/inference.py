import numpy as np
import torch
import soundfile as sf
import scipy.signal
import json
import os
import torch.nn as nn
import torch.nn.functional as F
from mir_eval.separation import bss_eval_sources

# --- 1. Load Config ---
if not os.path.exists("config.json"): raise FileNotFoundError("Config missing")
with open("config.json", "r") as f: CONF = json.load(f)

FS = CONF["fs"]
N_FFT = CONF["n_fft"]
HOP = CONF["hop_len"]
WIN_SIZE_SAMPLES = CONF["train_seg_samples"] # 32000
D = CONF["d"]
C = CONF["c"]
ANGLE_TARGET = 90.0
SIGMA = 1e-5
N_MICS = 2

# --- 2. Architecture (Must Match Training) ---
class FreqPreservingUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.enc1 = self._conv(2, 32)
        self.enc2 = self._conv(32, 64)
        self.enc3 = self._conv(64, 128)
        self.bot = self._conv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.dec3 = self._conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.dec2 = self._conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2))
        self.dec1 = self._conv(64, 32)
        self.out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def _conv(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()
        )
    
    def _match(self, x, target):
        if x.shape[3] != target.shape[3]:
            x = F.interpolate(x, size=target.shape[2:], mode='nearest')
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        u3 = self._match(self.up3(b), e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1).squeeze(1)

# --- 3. Physics Helper ---
def get_steering_vector(angle_deg, f, d, c):
    theta = np.deg2rad(angle_deg)
    tau1 = (d / 2) * np.cos(0) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(0) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f
    return np.array([[np.exp(-1j * omega * tau1)], [np.exp(-1j * omega * tau2)]], dtype=complex)

def calculate_metrics_manual(output, target, interf):
    if len(target) == 0: return 0, 0
    output = output / (np.linalg.norm(output) + 1e-6)
    target = target / (np.linalg.norm(target) + 1e-6)
    interf = interf / (np.linalg.norm(interf) + 1e-6)
    
    p_t = np.sum(np.dot(output, target)**2)
    p_i = np.sum(np.dot(output, interf)**2) + 1e-10
    return 10 * np.log10(p_t/p_i), 10 * np.log10(p_t/p_i)

# --- 4. Sliding Window Processor ---
def process_chunk(y_chunk, model):
    """ Runs MVDR on a single 2.0s chunk """
    f, t, Y = scipy.signal.stft(y_chunk.T, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)
    mag = np.abs(Y)
    ipd = np.angle(Y[0]) - np.angle(Y[1])
    
    X = torch.from_numpy(np.stack([np.log(mag[0] + 1e-7), ipd], axis=0)).float().unsqueeze(0)
    
    with torch.no_grad():
        Mask = model(X).squeeze(0).numpy()
    
    Mask_N = 1.0 - Mask
    S_out = np.zeros_like(Y[0], dtype=complex)
    
    for i in range(Y.shape[1]):
        if f[i] < 100: continue
        m_vec = Mask_N[i, :]
        Y_vec = Y[:, i, :]
        
        R_n = (Y_vec * np.sqrt(m_vec)) @ (Y_vec * np.sqrt(m_vec)).conj().T
        R_n = R_n / (np.sum(m_vec) + 1e-6) + SIGMA * np.eye(N_MICS)
        d = get_steering_vector(ANGLE_TARGET, f[i], D, C)
        try:
            w = np.linalg.solve(R_n, d)
            w /= (d.conj().T @ w + 1e-10)
        except: w = np.array([[1], [0]])
        S_out[i, :] = w.conj().T @ Y_vec

    S_final = S_out * np.maximum(Mask, 0.05)
    _, out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)
    return out

def main_deploy(input_path):
    print(f"Processing {input_path}...")
    if not os.path.exists("mask_estimator.pth"):
        print("Model not found.")
        return

    y_full, fs = sf.read(input_path, dtype='float32')
    WIN_SIZE = WIN_SIZE_SAMPLES 
    HOP = WIN_SIZE // 2            
    
    model = FreqPreservingUNet()
    model.load_state_dict(torch.load("mask_estimator.pth", map_location='cpu'))
    model.eval()
    
    # Overlap-Add Buffer
    output_buffer = np.zeros(len(y_full) + WIN_SIZE)
    norm_buffer = np.zeros(len(y_full) + WIN_SIZE)
    num_chunks = int(np.ceil(len(y_full) / HOP))
    
    print(f"Audio Length: {len(y_full)/FS:.2f}s. Processing {num_chunks} sliding windows...")

    for i in range(num_chunks):
        start = i * HOP
        end = start + WIN_SIZE
        chunk = y_full[start:end]
        
        if len(chunk) < WIN_SIZE:
            chunk = np.pad(chunk, ((0, WIN_SIZE - len(chunk)), (0, 0)))
            
        out_chunk = process_chunk(chunk, model)
        
        L = min(len(out_chunk), WIN_SIZE)
        output_buffer[start:start+L] += out_chunk[:L]
        norm_buffer[start:start+L] += 1.0

    norm_buffer[norm_buffer == 0] = 1.0
    final_output = output_buffer[:len(y_full)] / norm_buffer[:len(y_full)]
    
    out_name = f"enhanced_{os.path.basename(input_path)}"
    sf.write(out_name, final_output, fs)
    print(f"Saved: {out_name}")

    if "mixture_" in input_path:
        tgt, _ = sf.read("target_ref_TEST.wav")
        intf, _ = sf.read("interf_ref_TEST.wav")
        L = min(len(final_output), len(tgt))
        sir, _ = calculate_metrics_manual(final_output[:L], tgt[:L], intf[:L])
        print(f"SIR Improvement: {sir:.2f} dB")

if __name__ == "__main__":
    # Change this to your arbitrary file path
    INPUT_FILE = "mixture_TRAIN.wav" 
    if os.path.exists(INPUT_FILE):
        main_deploy(INPUT_FILE)
    else:
        print("Input file not found.")