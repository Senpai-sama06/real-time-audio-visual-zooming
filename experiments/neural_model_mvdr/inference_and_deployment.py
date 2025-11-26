import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import os
import scipy.signal
import librosa
from mir_eval.separation import bss_eval_sources
import warnings

# Suppress warnings that clutter the console
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. GLOBAL CONFIGURATION (MUST MATCH TRAINING) ---
FS = 16000          # Sample rate (Target for all input audio)
D = 0.04            # Mic spacing
C = 343.0           # Speed of sound
ANGLE_TARGET = 90.0 # Direction of focus (Where the MVDR aims)
N_FFT = 512
N_HOP = 256
N_MICS = 2
SEGMENT_LEN_SAMPLES = int(2.0 * FS) # 2.0 seconds (fixed input size for NN)
SIGMA = 1e-5        # Diagonal loading (Zoom knob setting for filtering)

# --- 2. ARCHITECTURE DEFINITION (Shallow CNN) ---

class ShallowCNNMaskEstimator(nn.Module):
    """
    Defines the Neural Network architecture. 
    This must exactly match the model used to create 'mask_estimator.pth'.
    """
    def __init__(self, n_freqs, n_time_frames, n_features=2):
        super(ShallowCNNMaskEstimator, self).__init__()
        self.n_freqs = n_freqs
        self.n_time_frames = n_time_frames
        
        # Sequential CNN structure 
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

# --- 3. PHYSICS & UTILITY FUNCTIONS ---

def get_steering_vector(angle_deg, f, d, c):
    """Calculates the steering vector for the target angle (90 deg)."""
    theta_rad = np.deg2rad(angle_deg)
    phi_rad = 0.0 
    tau_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi) / c
    omega = 2 * np.pi * f
    d_vec = np.array([[np.exp(-1j * omega * tau_m1)], [np.exp(-1j * omega * tau_m2)]], dtype=complex)
    return d_vec

def calculate_metrics_manual(output_signal, target_ref, interf_ref):
    # Function included for code completeness, but not used in external inference mode.
    output_signal = output_signal / np.linalg.norm(output_signal)
    target_ref = target_ref / np.linalg.norm(target_ref)
    interf_ref = interf_ref / np.linalg.norm(interf_ref)
    
    power_target = np.sum(np.dot(output_signal, target_ref)**2)
    power_interf = np.sum(np.dot(output_signal, interf_ref)**2) + 1e-10
    
    sir = 10 * np.log10(power_target / power_interf)
    sdr = bss_eval_sources(np.array([target_ref]), np.array([output_signal]))[0][0]
    
    return sdr, sir

def process_input_audio(y_mix_full, current_fs, target_fs, target_len):
    """
    Ensures input audio meets all technical requirements: 16kHz, Stereo, Fixed Length.
    """
    # 1. Resampling Check (Fixes FS if needed)
    if current_fs != target_fs:
        print(f"Warning: Resampling from {current_fs}Hz to {target_fs}Hz.")
        y_mix_full = librosa.resample(y_mix_full, orig_sr=current_fs, target_sr=target_fs)
        current_fs = target_fs
    
    # 2. Channel Check (Fixes Mono to Pseudo-Stereo)
    if y_mix_full.ndim == 1:
        # Mono file: Duplicate channel to create pseudo-stereo
        print("Warning: Input is Mono. Duplicating channel for 2-mic array.")
        y_mix_full = np.stack([y_mix_full, y_mix_full], axis=1)
    
    # Trim or Pad to fixed length for the NN
    if y_mix_full.shape[0] < target_len:
        print(f"Warning: Padding to {target_len/target_fs:.2f}s.")
        y_mix_full = np.pad(y_mix_full, ((0, target_len - y_mix_full.shape[0]), (0, 0)))
    elif y_mix_full.shape[0] > target_len:
        print(f"Warning: Trimming to {target_len/target_fs:.2f}s.")
        y_mix_full = y_mix_full[:target_len, :]

    # Transpose to [Channels, Samples] for STFT processing
    return y_mix_full.T 

# --- 4. MAIN DEPLOYMENT PIPELINE ---

def deploy_and_process(input_file_path):
    print(f"\n--- 1. Neural MVDR Processing: {os.path.basename(input_file_path)} ---")
    
    # --- Critical Checkpoint ---
    if not os.path.exists('mask_estimator.pth'):
        print("CRITICAL: 'mask_estimator.pth' missing. Ensure the model file is in this directory.")
        return

    # 1. Load Data
    try:
        y_mix_full, current_fs = sf.read(input_file_path, dtype='float32')
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    # 2. Pre-process the audio to match required format
    y_mix = process_input_audio(y_mix_full, current_fs, FS, SEGMENT_LEN_SAMPLES)
    
    # 3. STFT and Feature Prep
    f, t, Y_stft = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    n_channels, n_freqs, n_frames = Y_stft.shape
    
    # 4. Load Model and Prepare Input Features
    model = ShallowCNNMaskEstimator(n_freqs, n_frames) 
    model.load_state_dict(torch.load('mask_estimator.pth', map_location=torch.device('cpu'))) 
    model.eval()

    # Feature Calculation (Log-Mag and IPD)
    mag = np.abs(Y_stft)
    ipd = np.angle(Y_stft[0, :, :]) - np.angle(Y_stft[1, :, :])
    model_input = np.stack([np.log(mag[0, :, :] + 1e-7), ipd], axis=-1)
    
    X_deploy = torch.from_numpy(model_input).float().permute(2, 0, 1).unsqueeze(0) 

    # 5. Neural Inference (Get the Predicted Mask)
    with torch.no_grad():
        M_target_pred = model(X_deploy).squeeze(0).numpy()
        
    M_noise = 1.0 - M_target_pred # Noise Mask

    # 6. MVDR Core Logic
    S_mvdr_stft = np.zeros((n_freqs, n_frames), dtype=complex)
    
    for f_idx in range(n_freqs):
        f_val = f[f_idx]
        if f_val < 100: continue 
        
        m_f = M_noise[f_idx, :] 
        Y_f = Y_stft[:, f_idx, :] 
        
        # Masked Noise Covariance Matrix (R_noise)
        R_noise = (Y_f * np.sqrt(m_f) @ (Y_f * np.sqrt(m_f)).conj().T) / (np.sum(m_f) + 1e-6)
        
        R_loaded = R_noise + SIGMA * np.eye(N_MICS)
        d = get_steering_vector(ANGLE_TARGET, f_val, D, C)
        
        # Calculate MVDR Weights (w)
        try:
            w = np.linalg.solve(R_loaded, d)
            w /= (d.conj().T @ w + 1e-10)
        except:
            w = np.array([[1], [0]]) # Fallback to Mic 1
            
        S_mvdr_stft[f_idx, :] = w.conj().T @ Y_stft[:, f_idx, :]

    # 7. Post-Filter and iSTFT
    FLOOR = 0.05 
    gain_filter = np.maximum(M_target_pred, FLOOR) 
    S_final_stft = S_mvdr_stft * gain_filter
        
    _, s_out = scipy.signal.istft(S_final_stft, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-6)
    
    # 8. Save Output
    base_name = os.path.basename(input_file_path).split('.')[0]
    output_filename = f"output_neural_mvdr_{base_name}_ENHANCED.wav"
    sf.write(output_filename, s_out, FS)

    print(f"\nProcessing complete.")
    print(f"Enhanced audio saved to '{output_filename}'")
    print(f"NOTE: Target is assumed to be at {ANGLE_TARGET}° (Broadside).")


if __name__ == "__main__":
    
    # --- USER-DEFINED INPUT PATH ---
    # CHANGE THIS VARIABLE to the path of the WAV file you want to process.
    # The file will be automatically converted to 16kHz and stereo.
    INPUT_AUDIO_PATH = "" 
    
    if os.path.exists(INPUT_AUDIO_PATH):
        deploy_and_process(INPUT_AUDIO_PATH)
    else:
        print(f"\nCRITICAL ERROR: Input file not found at '{INPUT_AUDIO_PATH}'.")
        print("Please ensure the file exists and the path is correct.")