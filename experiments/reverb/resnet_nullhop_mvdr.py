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
# Note: Ensure config.json exists in the same directory
if not os.path.exists("config.json"):
    # Fallback defaults for safety if testing without file
    CONF = {
        "fs": 16000,
        "n_fft": 512,
        "hop_len": 128,
        "train_seg_samples": 32000,
        "d": 0.08,  # 8 cm
        "c": 343.0
    }
    print("Warning: config.json not found, using defaults.")
else:
    with open("config.json", "r") as f:
        CONF = json.load(f)

FS = CONF["fs"]
N_FFT = CONF["n_fft"]
HOP = CONF["hop_len"]
WIN_SIZE_SAMPLES = CONF["train_seg_samples"]
D = CONF["d"]
C = CONF["c"]
ANGLE_TARGET = 90.0
N_MICS = 2

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    """Calculates the theoretical steering vector for a given angle and freq"""
    theta = np.deg2rad(angle_deg)
    # Assuming Mic 1 is at d/2 and Mic 2 is at -d/2 (Linear Array centered at 0)
    # Or Mic 1 at 0 and Mic 2 at d. 
    # Standard array manifold for 2 mics dist d: [1, exp(-j*w*tau)]
    # Here using the definition from your snippet (centered array?):
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c # cos(theta-pi) = -cos(theta)
    
    omega = 2 * np.pi * f
    v = np.array([[np.exp(-1j * omega * tau1)], [np.exp(-1j * omega * tau2)]])
    
    # Normalize to reference mic (Mic 0) for phase consistency
    v = v / (v[0] + 1e-10)
    return v

# =====================================================================
# 4. PROCESS A 2-SECOND WINDOW (Hybrid Hard-Null)
# =====================================================================
# =====================================================================
# 4. PROCESS A 2-SECOND WINDOW (Hybrid Hard-Null)
# =====================================================================
def process_chunk(y_chunk, model, chunk_idx):
    """
    Runs Hybrid Beamforming:
    1. DL Mask Estimation
    2. Spatial Hard-Nulling of Interference (using EVD)
    3. Spectral Post-Filtering of Noise
    """
    print(f"\n--- Processing chunk {chunk_idx} ---")

    # STFT
    f, t, Y = scipy.signal.stft(y_chunk.T, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP)
    # Y shape: (M, F, T)

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
    Mask_Int = 1.0 - Mask # Interference probability

    # ------------------ HYBRID BEAMFORMER (HARD NULL) -----------------------
    tic = time.time()

    S_out = np.zeros_like(Y[0], dtype=complex)
    
    # Pre-allocate response vector [Target=1, Null=0]
    desired_response = np.array([[1], [0]], dtype=np.complex64)

    for i in range(Y.shape[1]): # Iterate over Frequencies
        
        # 1. Low Frequency Bypass
        # As discussed, no spatial resolution below ~200Hz for 8cm.
        # Just pass the Reference Mic to avoid noise explosion.
        if f[i] < 200: 
            S_out[i, :] = Y[0, i, :]
            continue

        # 2. Get Data Vectors
        m_int = Mask_Int[i, :] # Interference Weights (1, T)
        Y_vec = Y[:, i, :]     # Mic Signals (2, T)

        # 3. Estimate Interference Steering Vector (Data-Driven)
        # Calculate Interference Covariance Matrix
        # R_int = E[ Y * Y^H * Mask_Int ]
        # Weighted outer product sum
        R_int = (Y_vec * m_int) @ (Y_vec.conj().T)
        R_int /= (np.sum(m_int) + 1e-6) # Normalize

        # EVD to find dominant interference direction
        eigvals, eigvecs = np.linalg.eigh(R_int)
        v_int = eigvecs[:, -1].reshape(2, 1) # Principal eigenvector

        # Phase Normalize v_int (Reference Mic 0 = Real Positive)
        # This removes arbitrary phase shifts from EVD
        v_int = v_int / (v_int[0] / (np.abs(v_int[0]) + 1e-10))

        # 4. Get Target Steering Vector (Fixed Geometry as requested)
        # Pass global C (Speed of Sound) here
        v_tgt = get_steering_vector(ANGLE_TARGET, f[i], D, C)
        
        # 5. Formulate Constraint Matrix C_mat (Renamed from C)
        # We want: w^H * v_tgt = 1
        #          w^H * v_int = 0
        C_mat = np.column_stack((v_tgt, v_int)) # Shape (2, 2)

        # 6. Solve for Weights (Geometric Inversion)
        # w = (C_mat^H)^-1 * [1, 0]
        
        # Check Condition Number (Are Target and Jammer too close?)
        cond_num = np.linalg.cond(C_mat)
        
        if cond_num > 10: # Threshold for "Too Close"
            # Fallback: Standard Delay-and-Sum towards Target
            w = v_tgt / N_MICS
        else:
            try:
                # Solve C_mat^H * w = desired
                w = np.linalg.solve(C_mat.conj().T, desired_response)
            except np.linalg.LinAlgError:
                w = v_tgt / N_MICS

        # 7. Apply Spatial Filter
        # w shape (2,1), Y_vec shape (2, T) -> (1, T)
        S_out[i, :] = (w.conj().T @ Y_vec).squeeze()

    t_mvdr = time.time() - tic
    print(f"Hybrid BF Processing Time: {t_mvdr*1000:.2f} ms")

    # ------------------ POST-FILTER (SPECTRAL) -----------------------
    # Step 2 of the Two-Stage cleaning: Remove the diffuse noise (AWGN)
    # using the Soft Mask from the Neural Network.
    
    # Optional: Soft-threshold the mask to prevent musical noise
    Mask_Soft = np.maximum(Mask, 0.05) 
    
    S_final = S_out * Mask_Soft
    
    # ISTFT
    _, out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP)

    return out, t_infer, t_mvdr

# =====================================================================
# 5. MAIN DEPLOY
# =====================================================================
def main_deploy(input_path):

    print(f"\nProcessing {input_path} ...")

    if not os.path.exists("mask_estimator.pth"):
        print("Model 'mask_estimator.pth' not found.")
        return

    # Load audio
    y_full, fs = sf.read(input_path, dtype='float32')
    # If mono, duplicate channels for testing (though logic requires stereo)
    if y_full.ndim == 1:
        print("Error: Input audio is mono. This pipeline requires 2-channel audio.")
        return
        
    WIN_SIZE = WIN_SIZE_SAMPLES
    HOP_WIN = WIN_SIZE // 2

    # Load model
    model = DeepFPU().to(device)
    try:
        model.load_state_dict(torch.load("mask_estimator.pth", map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval()

    output_buffer = np.zeros(len(y_full) + WIN_SIZE)
    norm_buffer = np.zeros(len(y_full) + WIN_SIZE)

    num_chunks = int(np.ceil(len(y_full) / HOP_WIN))
    print(f"Audio Length: {len(y_full)/FS:.2f}s | Chunks: {num_chunks}")

    total_infer_time = 0
    total_mvdr_time = 0
    total_start = time.time()

    for i in range(num_chunks):
        start = i * HOP_WIN
        end = start + WIN_SIZE

        chunk = y_full[start:end]
        # Pad if last chunk is too short
        if len(chunk) < WIN_SIZE:
            chunk = np.pad(chunk, ((0, WIN_SIZE - len(chunk)), (0, 0)))

        out_chunk, t_inf, t_mvdr = process_chunk(chunk, model, i)

        total_infer_time += t_inf
        total_mvdr_time += t_mvdr

        L = min(len(out_chunk), WIN_SIZE)
        # OLA (Overlap-Add)
        # Note: The output buffer is 1D (Mono)
        output_buffer[start:start+L] += out_chunk[:L]
        norm_buffer[start:start+L] += 1.0

    total_time = time.time() - total_start

    # Normalize OLA
    norm_buffer[norm_buffer == 0] = 1.0
    final_output = output_buffer[:len(y_full)] / norm_buffer[:len(y_full)]

    out_name = f"hybrid_null_{os.path.basename(input_path)}"
    sf.write(out_name, final_output, fs)

    print("\n================ Timing Summary ================")
    print(f"Total Mask Estimation Time: {total_infer_time:.3f} s")
    print(f"Total Beamforming Time:     {total_mvdr_time:.3f} s")
    print(f"Total End-to-End Time:      {total_time:.3f} s")
    print("================================================")
    print(f"Saved output: {out_name}")

# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    INPUT_FILE = "sample/mixture.wav"
    if os.path.exists(INPUT_FILE):
        main_deploy(INPUT_FILE)
    else:
        print(f"Input file '{INPUT_FILE}' not found.")