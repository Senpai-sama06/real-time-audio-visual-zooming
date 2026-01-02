import numpy as np
import torch
import soundfile as sf
import scipy.signal
import scipy.linalg
from scipy.optimize import root_scalar
import json
import os
import time
import torch.nn as nn
import torch.nn.functional as F
# from mir_eval.separation import bss_eval_sources # Optional if you have it installed

# =====================================================================
# 1. LOAD CONFIG & CONSTANTS
# =====================================================================
try:
    FS = 16000
    N_FFT = 1024
    HOP = 512
    WIN_SIZE_SAMPLES = 32000
    D = 0.04
    C = 343.0
except:
    pass


# BEAMFORMING TARGETS
ANGLE_TARGET = 90.0
SIGMA = 1e-5
N_MICS = 2

# The audio within [ANGLE_TARGET - theta/2, ANGLE_TARGET + theta/2] 
THETA = 40.0 

# =====================================================================
# 2. MODEL ARCHITECTURE (DeepFPU - Unchanged)
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
        self.enc1_conv = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc2_conv = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64))
        self.enc3_conv = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128))
        self.enc4_conv = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256))
        self.bottleneck = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), ResBlock(512), ResBlock(512))
        self.up4 = nn.ConvTranspose2d(512, 256, (1, 2), stride=(1, 2))
        self.dec4_conv = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256))
        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.dec3_conv = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128))
        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.dec2_conv = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64))
        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2))
        self.dec1_conv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
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

# =====================================================================
# 3. PHYSICS & RMVB SOLVER
# =====================================================================
def get_steering_vector(angle_deg, f, d, c):
    theta = np.deg2rad(angle_deg)
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f
    return np.array([[np.exp(-1j * omega * tau1)], [np.exp(-1j * omega * tau2)]])

def robust_beamformer_solver(R_n, a_nom, epsilon):
    """
    Computes Robust Minimum Variance Beamforming weights using Lagrange multipliers.
    """
    N = R_n.shape[0]
    R_real = R_n.real
    R_imag = R_n.imag
    R_big = np.block([[R_real, -R_imag], [R_imag,  R_real]])
    c = np.concatenate((a_nom.real, a_nom.imag)).reshape(-1)
    A_big = epsilon * np.eye(2 * N)
    Q = (A_big @ A_big.T) - np.outer(c, c)

    try:
        vals, V = scipy.linalg.eigh(Q, R_big)
    except np.linalg.LinAlgError:
        return np.linalg.solve(R_n, a_nom) / (a_nom.conj().T @ np.linalg.solve(R_n, a_nom))

    c_bar = V.T @ c

    def secular_eq(lam):
        denom = (1 + lam * vals)
        if np.any(np.abs(denom) < 1e-9): return 1e5 
        term1 = (lam**2) * np.sum((c_bar**2 * vals) / (denom**2))
        term2 = -2 * lam * np.sum((c_bar**2) / denom)
        return term1 + term2 - 1

    lambda_opt = 0.0
    try:
        sol = root_scalar(secular_eq, bracket=[0, 1e5], method='brentq')
        lambda_opt = sol.root
    except ValueError:
        pass

    scaling = -lambda_opt / (1 + lambda_opt * vals)
    x_opt = V @ (scaling * c_bar)
    w_rob = x_opt[:N] + 1j * x_opt[N:]
    return w_rob.reshape(-1, 1)

def calculate_metrics_manual(output, target, interf):
    if len(target) == 0: return 0, 0
    output = output / (np.linalg.norm(output) + 1e-6)
    target = target / (np.linalg.norm(target) + 1e-6)
    interf = interf / (np.linalg.norm(interf) + 1e-6)
    p_t = np.sum(np.dot(output, target)**2)
    p_i = np.sum(np.dot(output, interf)**2) + 1e-10
    sir_val = 10 * np.log10(p_t/p_i)
    return sir_val, sir_val

# --- NEW: BEAM METRIC CALCULATION ---
def compute_broadband_metrics(W_matrix, freq_bins, d, c, angle_res=0.5):
    """
    Calculates Peak Angle and -3dB Beam Width from the array response.
    """
    angles = np.arange(0, 181, angle_res)
    theta_rad = np.deg2rad(angles)
    
    # Filter for Speech Frequencies (300Hz - 3400Hz)
    valid_idx = np.where((freq_bins >= 300) & (freq_bins <= 3400))[0]
    
    if len(valid_idx) == 0:
        return 0.0, 0.0 
        
    W_sub = W_matrix[valid_idx, :]   
    f_sub = freq_bins[valid_idx]     
    
    omega = 2 * np.pi * f_sub[:, None] 
    cos_theta = np.cos(theta_rad)[None, :] 
    
    tau1 = (d/2) * cos_theta / c
    tau2 = -(d/2) * cos_theta / c 
    
    sv = np.stack([
        np.exp(-1j * omega * tau1), 
        np.exp(-1j * omega * tau2)
    ], axis=-1)
    
    w_conj = W_sub.conj()[:, None, :]
    beam_response_f = np.abs(np.sum(w_conj * sv, axis=-1))**2
    
    # Average broadband response 
    total_response = np.mean(beam_response_f, axis=0)
    
    # Normalize to 1.0 (Peak)
    max_val = np.max(total_response) + 1e-9
    total_response /= max_val
    
    # 1. Find Peak Angle
    peak_idx = np.argmax(total_response)
    peak_angle = angles[peak_idx]
    
    # 2. Find -3dB Beam Width (where power >= 0.5)
    above_thresh = np.where(total_response >= 0.5)[0]
    
    if len(above_thresh) > 0:
        width_3db = angles[above_thresh[-1]] - angles[above_thresh[0]]
    else:
        width_3db = 0.0
    
    return peak_angle, width_3db

# =====================================================================
# 4. PROCESS A 2-SECOND WINDOW (MODIFIED)
# =====================================================================
def process_chunk(y_chunk, model, chunk_idx, device):  # Added device argument
    # STFT
    f_vec, t, Y = scipy.signal.stft(y_chunk.T, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP)
    mag = np.abs(Y)
    
    if Y.shape[-1] < 2: 
        return np.zeros_like(y_chunk), 0, 0, 0, 0

    ipd = np.angle(Y[0]) - np.angle(Y[1])

    X = torch.from_numpy(np.stack([
        np.log(mag[0] + 1e-7),
        ipd
    ], axis=0)).float().unsqueeze(0).to(device)

    # ------------------ MODEL INFERENCE -----------------------
    tic = time.time()
    with torch.no_grad():
        Mask = model(X).squeeze(0)
    
    # Now this works because 'device' is a torch.device object
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    t_infer = time.time() - tic

    Mask = Mask.cpu().numpy()
    Mask_N = 1 - Mask 

    # ------------------ RMVB PROCESSING -----------------------
    tic = time.time()
    S_out = np.zeros_like(Y[0], dtype=complex)
    W_storage = np.zeros((Y.shape[1], N_MICS), dtype=complex)

    for i in range(Y.shape[1]):
        # Skip DC or very low freqs to avoid numerical instability
        if f_vec[i] < 100:
            S_out[i, :] = Y[0, i, :]
            W_storage[i, :] = np.array([1, 0])
            continue

        m_vec = Mask_N[i, :]
        Y_vec = Y[:, i, :]
        
        # Estimate R_n (Noise Covariance)
        Y_weighted = Y_vec * np.sqrt(m_vec)
        R_n = Y_weighted @ Y_weighted.conj().T
        R_n = R_n / (np.sum(m_vec) + 1e-6) + SIGMA * np.eye(N_MICS)

        # 1. Get Nominal Steering Vector (Center of Beam)
        d_nom = get_steering_vector(ANGLE_TARGET, f_vec[i], D, C)

        # --- NEW LOGIC START: CALCULATE EPSILON FROM THETA ---
        if THETA > 0:
            # 2. Get Edge Steering Vector (Limit of desired view)
            d_edge = get_steering_vector(ANGLE_TARGET + (THETA / 2.0), f_vec[i], D, C)
            
            # 3. Set Epsilon as the distance between Center and Edge
            epsilon_adaptive = np.linalg.norm(d_edge - d_nom)
        else:
            epsilon_adaptive = 0.0
        # --- NEW LOGIC END ---

        # Compute RMVB Weights
        if epsilon_adaptive > 1e-9:
            w = robust_beamformer_solver(R_n, d_nom, epsilon_adaptive)
        else:
            try:
                w = np.linalg.solve(R_n, d_nom)
                w = w / (d_nom.conj().T @ w + 1e-10)
            except:
                w = np.array([[1], [0]])

        S_out[i, :] = w.conj().T @ Y_vec
        W_storage[i, :] = w.flatten()
    t_beam = time.time() - tic
    
    # Calculate metrics
    pk, width = compute_broadband_metrics(W_storage, f_vec, D, C)

    # ISTFT
    S_final = S_out * np.maximum(Mask, 0.05)
    _, out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_FFT - HOP)

    return out, t_infer, t_beam, pk, width

# =====================================================================
# 5. MAIN DEPLOY
# =====================================================================
def main_deploy(input_path, FOLDER):
    print(f"\nProcessing {input_path} ...")

    if not os.path.exists("mask_estimator.pth"):
        print("Warning: 'mask_estimator.pth' not found. Creating a random one.")
        temp_model = DeepFPU()
        torch.save(temp_model.state_dict(), "mask_estimator.pth")

    y_full, fs = sf.read(input_path, dtype='float32')
    if len(y_full.shape) == 1:
        y_full = np.stack([y_full, y_full], axis=1)

    WIN_SIZE = WIN_SIZE_SAMPLES
    HOP_SIZE = WIN_SIZE // 2
    
    # --- CORRECTED DEVICE DEFINITION ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    model = DeepFPU().to(device)
    try:
        model.load_state_dict(torch.load("mask_estimator.pth", map_location=device))
    except:
        print("Model mismatch or load error. Ensure .pth matches architecture.")
    model.eval()

    output_buffer = np.zeros(len(y_full) + WIN_SIZE)
    norm_buffer = np.zeros(len(y_full) + WIN_SIZE)
    
    num_chunks = int(np.ceil(len(y_full) / HOP_SIZE))
    print(f"Audio Length: {len(y_full)/FS:.2f}s | Chunks: {num_chunks}")

    total_infer_time = 0
    total_beam_time = 0
    total_start = time.time()
    
    angles_list = []
    widths_list = []

    for i in range(num_chunks):
        start = i * HOP_SIZE
        end = start + WIN_SIZE

        chunk = y_full[start:end]
        if len(chunk) < WIN_SIZE:
             pad_amt = WIN_SIZE - len(chunk)
             chunk = np.pad(chunk, ((0, pad_amt), (0, 0)))

        out_chunk, t_inf, t_beam, pk, wd = process_chunk(chunk, model, i, device)
        
        angles_list.append(pk)
        widths_list.append(wd)
        
        total_infer_time += t_inf
        total_beam_time += t_beam

        L = min(len(out_chunk), WIN_SIZE)
        output_buffer[start:start+L] += out_chunk[:L]
        norm_buffer[start:start+L] += 1.0

    total_time = time.time() - total_start

    norm_buffer[norm_buffer == 0] = 1.0
    final_output = output_buffer[:len(y_full)] / norm_buffer[:len(y_full)]

    out_name = f"{FOLDER}/enhanced_RMVB__Switch_Mix_{os.path.basename(input_path)}"
    sf.write(out_name, final_output, fs)

    print("\n================ Processing Summary ================")
    print(f"Total RMVB Time:            {total_beam_time:.3f} s")
    print("-" * 40)
    
    avg_angle = np.mean(angles_list)
    avg_width = np.mean(widths_list)
    
    print(f"Average Look Angle:         {avg_angle:.1f} degrees")
    print(f"Average Beam Width (-3dB):  {avg_width:.2f} degrees")
    print("====================================================")
    print(f"Saved output: {out_name}")

    if "mixture_" in input_path:
        try:
            print("\nCalculating Metrics...")
            tgt_path = "target_ref_TEST.wav"
            intf_path = "interf_ref_TEST.wav"
            
            if os.path.exists(tgt_path) and os.path.exists(intf_path):
                tgt, _ = sf.read(tgt_path)
                intf, _ = sf.read(intf_path)
                L = min(len(final_output), len(tgt))
                sir, _ = calculate_metrics_manual(final_output[:L], tgt[:L], intf[:L])
                print(f"SIR Improvement: {sir:.2f} dB")
            else:
                print(f"Reference files {tgt_path} or {intf_path} not found.")
        except Exception as e:
            print(f"Error calculating SIR: {e}")

if __name__ == "__main__":
    HOME_DIR = "/home/communications-lab/ramakrishna/real-time-audio-visual-zooming/experiments/paper/arun"
    INPUT_FILE = "mixture_TEST_4.wav" 
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
    main_deploy(INPUT_FILE, HOME_DIR)