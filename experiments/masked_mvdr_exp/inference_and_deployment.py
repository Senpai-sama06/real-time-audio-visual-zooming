import numpy as np
import torch
import torch.nn as nn
import scipy.signal
import soundfile as sf
import os
from mir_eval.separation import bss_eval_sources
import warnings

# Suppress warnings that clutter the console
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Constants (Must Match Training) ---
FS = 16000
D = 0.04
C = 343.0
ANGLE_TARGET = 90.0
N_FFT = 512
N_HOP = 256
N_MICS = 2

# CRITICAL FIX: Match the segment length used in training (2.0 seconds)
SEGMENT_LEN_SAMPLES = int(2.0 * FS) 
store_dir = "/home/cse-sdpl/paarth/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/samples"
# MVDR Settings
SIGMA = 1e-5 

# --- 2. Re-Define Model Architecture (SHALLOW CNN - MATCHES TRAINER) ---
class ShallowCNNMaskEstimator(nn.Module):
    def __init__(self, n_freqs, n_time_frames, n_features=2):
        super(ShallowCNNMaskEstimator, self).__init__()
        self.n_freqs = n_freqs
        self.n_time_frames = n_time_frames
        
        # Sequential CNN structure (MATCHES the training file's architecture)
        self.model = nn.Sequential(
            # Input: [B, 2, Freq, Time]
            nn.Conv2d(n_features, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final 1x1 convolution maps 32 channels to 1 mask channel
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
            # Output: [B, 1, Freq, Time]
        )

    def forward(self, x):
        # x shape: [Batch, Features, Freq, Time]
        x = self.model(x)
        # Squeeze channel dimension to match target shape [Batch, Freq, Time]
        return x.squeeze(1)

# --- 3. Core MVDR and Feature Functions (Unchanged) ---
def get_steering_vector(angle_deg, f, d, c):
    """Calculates the steering vector for the target angle."""
    theta_rad = np.deg2rad(angle_deg)
    phi_rad = 0.0 
    tau_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi) / c
    omega = 2 * np.pi * f
    d_vec = np.array([[np.exp(-1j * omega * tau_m1)], [np.exp(-1j * omega * tau_m2)]], dtype=complex)
    return d_vec

def calculate_metrics_manual(output_signal, target_ref, interf_ref):
    """Manual SIR calculation using projection (robust for BSS_EVAL issues)."""
    output_signal = output_signal / np.linalg.norm(output_signal)
    target_ref = target_ref / np.linalg.norm(target_ref)
    interf_ref = interf_ref / np.linalg.norm(interf_ref)
    
    alpha = np.dot(output_signal, target_ref)
    e_target = alpha * target_ref
    beta = np.dot(output_signal, interf_ref)
    e_interf = beta * interf_ref
    
    power_target = np.sum(e_target**2)
    power_interf = np.sum(e_interf**2) + 1e-10
    
    sir = 10 * np.log10(power_target / power_interf)
    sdr = bss_eval_sources(np.array([target_ref]), np.array([output_signal]))[0][0] # Fallback to mir_eval SDR
    
    return sdr, sir

# --- 4. Main Deployment Pipeline ---
def deploy_and_validate():
    print("--- 3. Neural MVDR Deployment and Validation (SHALLOW CNN) ---")
    
    # 1. Load Data
    input_file = f"{store_dir}/mixture_3_sources.wav"
    if not os.path.exists(input_file):
        print("CRITICAL: Mixture file missing. Run world.py first.")
        return

    # Load and immediately trim ALL files to the segment length used in training (2.0s)
    y_mix_full, fs = sf.read(input_file, dtype='float32')
    
    # --- CRITICAL TRIM ---
    SEGMENT_LEN_SAMPLES = int(2.0 * FS) 
    y_mix = y_mix_full[:SEGMENT_LEN_SAMPLES, :].T # Take only the first 2.0s
    s_tgt_ref, _ = sf.read(f"{store_dir}/target_reference.wav", dtype='float32')
    s_tgt_ref = s_tgt_ref[:SEGMENT_LEN_SAMPLES]
    s_int_ref, _ = sf.read(f"{store_dir}/interference_reference.wav", dtype='float32')
    s_int_ref = s_int_ref[:SEGMENT_LEN_SAMPLES]
    # --- END CRITICAL TRIM ---

    # 2. STFT
    f, t, Y_stft = scipy.signal.stft(y_mix, fs=fs, nperseg=N_FFT, noverlap=N_HOP)
    n_channels, n_freqs, n_frames = Y_stft.shape
    
    # 3. Load Trained Model
    # Initialize the SHALLOW CNN model with the exact dimensions of the STFT
    model = ShallowCNNMaskEstimator(n_freqs, n_frames) 
    if not os.path.exists("/home/cse-sdpl/paarth/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/mask_estimator_old.pth"):
        print("CRITICAL: Model file 'mask_estimator.pth' missing. Run train_neural_mask.py first.")
        return
        
    # NOTE: We load the model trained by the SHALLOW CNN
    model.load_state_dict(torch.load("/home/cse-sdpl/paarth/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/mask_estimator_old.pth"))
    model.eval()
    print("Model loaded successfully.")

    # 4. Prepare NN Input Features
    mag = np.abs(Y_stft)
    ipd = np.angle(Y_stft[0, :, :]) - np.angle(Y_stft[1, :, :])
    model_input = np.stack([np.log(mag[0, :, :] + 1e-7), ipd], axis=-1)
    
    # [Freq, Time, Features] -> [Features, Freq, Time] -> [Batch, Features, Freq, Time]
    X_deploy = torch.from_numpy(model_input).float().permute(2, 0, 1).unsqueeze(0) 

    # 5. NN Inference (Get the Predicted Mask)
    # The deployment model MUST run on the CPU if the training was interrupted
    device = torch.device("cpu") # Default to CPU for safe loading
    model.to(device)
    X_deploy = X_deploy.to(device)

    with torch.no_grad():
        M_target_pred = model(X_deploy).squeeze(0).numpy()
        
    M_noise = 1.0 - M_target_pred # Noise Mask

    # 6. MVDR Core Logic
    print(f"Running MVDR with predicted mask (Sigma={SIGMA:.1e})...")
    S_mvdr_stft = np.zeros((n_freqs, n_frames), dtype=complex)
    
    for f_idx in range(n_freqs):
        if f[f_idx] < 100: continue 
        
        # Calculate R_noise using the predicted mask
        m_f = M_noise[f_idx, :] 
        Y_f = Y_stft[:, f_idx, :] 
        Y_weighted = Y_f * np.sqrt(m_f) 
        R_noise = (Y_weighted @ Y_weighted.conj().T) / (np.sum(m_f) + 1e-6)
        
        # MVDR Weight calculation (The final spatial filter)
        R_loaded = R_noise + SIGMA * np.eye(N_MICS)
        d = get_steering_vector(ANGLE_TARGET, f[f_idx], D, C)
        
        try:
            w = np.linalg.solve(R_loaded, d)
            w /= (d.conj().T @ w + 1e-10)
        except:
            w = np.array([[1], [0]])
            
        S_mvdr_stft[f_idx, :] = w.conj().T @ Y_stft[:, f_idx, :]

    # 7. Post-Filter and iSTFT
    print("Applying Spectral Post-Filter...")
    
    # Apply soft post-filter using the TARGET mask
    FLOOR = 0.05 
    gain_filter = np.maximum(M_target_pred, FLOOR) 
    S_final_stft = S_mvdr_stft * gain_filter
        
    _, s_out = scipy.signal.istft(S_final_stft, fs=fs, nperseg=N_FFT, noverlap=N_HOP)
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-6)
    
    sf.write(f"{store_dir}/output_neural_mvdr.wav", s_out, fs)

    # 8. Validation 
    s_mix_mic1 = y_mix[0, :] # Get raw mixture from mic 1 (baseline)
    
    # Ensure all inputs to validation have the same length as the final output
    validation_len = s_out.shape[0]
    
    sdr_b, sir_b = calculate_metrics_manual(s_mix_mic1[:validation_len], s_tgt_ref[:validation_len], s_int_ref[:validation_len])
    sdr_m, sir_m = calculate_metrics_manual(s_out, s_tgt_ref[:validation_len], s_int_ref[:validation_len])

    print("\n\n=== NEURAL MVDR RESULTS ===")
    print(f"BASELINE (Raw Mic 1):      SIR: {sir_b:.2f} dB, SDR: {sdr_b:.2f} dB")
    print(f"NEURAL MVDR (Masked):      SIR: {sir_m:.2f} dB, SDR: {sdr_m:.2f} dB")
    print("=" * 40)
    print(f"SIR IMPROVEMENT: +{sir_m - sir_b:.2f} dB")
    print(f"File saved to 'output_neural_mvdr.wav'")


if __name__ == "__main__":
    deploy_and_validate()