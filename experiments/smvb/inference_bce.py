import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import scipy.signal
import os
import time
import tracemalloc
import sys
import json

# --- Config Helper ---
try:
    from src import config
    FS = config.FS
    N_FFT = config.N_FFT
    HOP_LEN = config.HOP_LEN
    WIN_SIZE = config.WIN_SIZE
    MIC_DIST = config.MIC_DIST
    C_SPEED = config.C_SPEED
    RESULTS_DIR = config.RESULTS_DIR
except ImportError:
    FS = 16000
    N_FFT = 1024
    HOP_LEN = 512
    WIN_SIZE = 32000
    MIC_DIST = 0.08
    C_SPEED = 343.0
    RESULTS_DIR = "results"

# --- Derived Constants ---
ANGLE_TARGET = 90.0
FREQ_BINS = (N_FFT // 2) + 1
N_MICS = 2
FIXED_TIME_STEPS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Model Architecture (Must match Training: 5 Channels)
# ==========================================

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
       
        # -- Encoder --
        # Layer 1: Input 5 channels -> 32
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        # Layer 2: 32 -> 64
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64)
        )
        # Layer 3: 64 -> 128
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128)
        )
        # Layer 4: 128 -> 256
        self.enc4_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256)
        )
       
        # -- Bottleneck --
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            ResBlock(512),
            ResBlock(512)
        )
       
        # -- Decoder --
        self.up4 = nn.ConvTranspose2d(512, 256, (1, 2), stride=(1, 2))
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64)
        )
        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2))
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
       
        self.out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

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

# ==========================================
# 2. Physics Helpers (Beamforming Logic)
# ==========================================

def get_steering_vector_single(f, angle_deg, d, c):
    theta = np.deg2rad(angle_deg)
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f
    v = np.array([[np.exp(-1j * omega * tau1)], [np.exp(-1j * omega * tau2)]])
    v = v / (v[0] + 1e-10)
    return v

def advanced_hybrid_bf(Y, mask, f_bins):
    n_freq = Y.shape[1]
    n_frames = Y.shape[2]
    S_out = np.zeros((n_freq, n_frames), dtype=complex)
   
    desired_response = np.array([[1], [0]], dtype=np.complex64)
    mask_int = 1.0 - mask
   
    LAMBDA_MIN = 1e-6
    LAMBDA_MAX = 0.1
    RANK_1_THRESH = 3.0
   
    for i in range(n_freq):
        f_hz = f_bins[i]
       
        if f_hz < 200:
            S_out[i, :] = Y[0, i, :]
            continue
           
        Y_vec = Y[:, i, :]
        m_int_vec = mask_int[i, :]
       
        denom_int = np.sum(m_int_vec) + 1e-6
        Y_weighted = Y_vec * np.sqrt(m_int_vec)
        Phi_int = (Y_weighted @ Y_weighted.conj().T) / denom_int
       
        R_yy = (Y_vec @ Y_vec.conj().T) / n_frames
       
        try:
            eigvals, eigvecs = np.linalg.eigh(Phi_int)
            lambda_1 = eigvals[1]
            lambda_2 = eigvals[0] + 1e-10
            ratio = lambda_1 / lambda_2
            v_int = eigvecs[:, 1].reshape(2, 1)
        except:
            ratio = 0
            v_int = np.zeros((2,1))

        v_tgt = get_steering_vector_single(f_hz, ANGLE_TARGET, MIC_DIST, C_SPEED)
       
        if ratio > RANK_1_THRESH:
            C_mat = np.column_stack((v_tgt, v_int))
            if np.linalg.cond(C_mat) > 10:
                 w = v_tgt / N_MICS
            else:
                try:
                    w = np.linalg.solve(C_mat.conj().T, desired_response)
                except np.linalg.LinAlgError:
                    w = v_tgt / N_MICS
        else:
            trace_R = np.trace(R_yy).real
            diag_load = max(LAMBDA_MIN, min(LAMBDA_MAX, 0.01 * trace_R))
            R_loaded = Phi_int + diag_load * np.eye(N_MICS)
            try:
                w_unnorm = np.linalg.solve(R_loaded, v_tgt)
                norm_factor = (v_tgt.conj().T @ w_unnorm).item()
                w = w_unnorm / (norm_factor + 1e-10)
            except:
                w = v_tgt / N_MICS
               
        S_out[i, :] = (w.conj().T @ Y_vec).squeeze()
       
    return S_out

# ==========================================
# 3. PyTorch Inference Wrapper
# ==========================================

class PyTorchBeamformer:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
           
        print(f"[PyTorch] Loading DeepFPU from {model_path} on {DEVICE}...")
        self.model = DeepFPU().to(DEVICE)
       
        state_dict = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)
        self.model.eval()
       
        self.freq_map = np.linspace(0, 1, FREQ_BINS, dtype=np.float32)[:, np.newaxis]

    def predict_mask(self, log_mag, raw_ipd, cross_power, auto_power):
        sin_ipd = np.sin(raw_ipd)
        cos_ipd = np.cos(raw_ipd)
        t_steps = log_mag.shape[1]
        f_map_tiled = np.tile(self.freq_map, (1, t_steps))
       
        # Calculate MSC (Magnitude Squared Coherence)
        msc = np.abs(cross_power) / (np.sqrt(np.abs(auto_power[0]) * np.abs(auto_power[1])) + 1e-9)

        # Stack 5 Channels: LogMag, Sin, Cos, Freq, MSC
        input_np = np.stack([log_mag, sin_ipd, cos_ipd, f_map_tiled, msc], axis=-1)
        input_np = np.transpose(input_np, (2, 0, 1))
        input_tensor = torch.from_numpy(input_np).float().unsqueeze(0).to(DEVICE)
       
        # Pad time dimension to be multiple of 16 (for pooling)
        pad_size = 16
        if t_steps % pad_size != 0:
            pad_w = pad_size - (t_steps % pad_size)
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, 0))
        else:
            pad_w = 0

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
           
        output_mask = output_tensor.squeeze().cpu().numpy()
        if pad_w > 0:
            output_mask = output_mask[:, :-pad_w]
           
        return output_mask

# ==========================================
# 4. Main Processing Logic
# ==========================================

def enhance_audio(run_name, input_path, model_path):
    result_dir = os.path.join(RESULTS_DIR, f"{run_name}_results")
    os.makedirs(result_dir, exist_ok=True)
   
    output_filename = f"{run_name}_enhanced.wav"
    output_path = os.path.join(result_dir, output_filename)

    print(f"\n[INF] --- Starting Inference ---")
    print(f"[INF] Input:  {input_path}")
    print(f"[INF] Model:  {model_path}")

    # Load Audio
    try:
        y, sr = sf.read(input_path, dtype='float32')
    except Exception as e:
        print(f"[Error] Could not read audio file: {e}")
        return

    if sr != FS:
        print(f"[Warning] SR mismatch. Input: {sr}, Config: {FS}. Audio quality may degrade.")
   
    if y.ndim == 1:
        print("[Error] Input is mono. Requires 2 channels (Stereo).")
        return

    # Buffers
    chunk_size = WIN_SIZE
    hop_size = chunk_size // 2
    out_buf = np.zeros(len(y))
    norm_buf = np.zeros(len(y))
   
    # Initialize Model
    try:
        bf = PyTorchBeamformer(model_path)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return
   
    num_chunks = int(np.ceil(len(y) / hop_size))
    start_time = time.time()
    t_acc = 0.0
   
    f_hz = np.fft.rfftfreq(N_FFT, 1.0/FS)
   
    tracemalloc.start()
   
    # Processing Loop
    print(f"[INF] Processing {num_chunks} chunks...")
    for i in range(num_chunks):
        start = i * hop_size
        end = start + chunk_size
        chunk = y[start:end]
       
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, ((0, chunk_size - len(chunk)), (0, 0)))
           
        f_bins, t_bins, Y = scipy.signal.stft(chunk.T, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP_LEN)
       
        mag = np.abs(Y)
        log_mag = np.log(mag[0] + 1e-7)
        ipd = np.angle(Y[0]) - np.angle(Y[1])
       
        # Prepare Power Spectra for MSC calculation
        cross_power = Y[0] * np.conj(Y[1])
        auto_power = [Y[0] * np.conj(Y[0]), Y[1] * np.conj(Y[1])]
       
        # 1. AI Mask
        mask = bf.predict_mask(log_mag, ipd, cross_power, auto_power)
       
        st = time.time()
       
        min_f = min(mask.shape[0], Y.shape[1])
        min_t = min(mask.shape[1], Y.shape[2])
        Y_trunc = Y[:, :min_f, :min_t]
        mask_trunc = mask[:min_f, :min_t]
       
        # 2. Physics Beamforming
        # S_out = advanced_hybrid_bf(Y_trunc, mask_trunc, f_hz[:min_f])
        S_out = Y_trunc[0] * mask_trunc 
        # 3. Post-Filter
        mask_soft = np.maximum(mask_trunc, 0.1)
        S_final = S_out * mask_soft
       
        _, chunk_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP_LEN)
       
        w_len = min(len(chunk_out), len(out_buf[start:]))
        out_buf[start:start+w_len] += chunk_out[:w_len]
        norm_buf[start:start+w_len] += 1.0
       
        t_acc += (time.time() - st)
       
        if i % 10 == 0:
            current_mem, _ = tracemalloc.get_traced_memory()
            sys.stdout.write(f"\r   > Chunk {i}/{num_chunks} | RAM: {current_mem/1024/1024:.1f} MB")
            sys.stdout.flush()

    tracemalloc.stop()
    print()
   
    total_time = time.time() - start_time
    print(f"[INF] Total Time: {total_time:.3f}s | BF Calculation: {t_acc:.3f}s")
       
    # Normalize & Save
    final = out_buf / np.maximum(norm_buf, 1.0)
    final = final / (np.max(np.abs(final)) + 1e-9)
   
    sf.write(output_path, final, FS)
    print(f"[SUCCESS] Enhanced Audio Saved: {output_path}")

# ==========================================
# 5. USER CONFIGURATION (KAGGLE RUN)
# ==========================================

INPUT_AUDIO_PATH = "sample/mixture.wav"
MODEL_FILE_PATH = "bce_model.pth"
RUN_NAME = "sample"

if __name__ == "__main__":
    if os.path.exists(INPUT_AUDIO_PATH) and os.path.exists(MODEL_FILE_PATH):
        enhance_audio(RUN_NAME, INPUT_AUDIO_PATH, MODEL_FILE_PATH)
    else:
        print("!!! PLEASE CHECK PATHS !!!")
        print(f"Input exists? {os.path.exists(INPUT_AUDIO_PATH)} -> {INPUT_AUDIO_PATH}")
        print(f"Model exists? {os.path.exists(MODEL_FILE_PATH)} -> {MODEL_FILE_PATH}")
