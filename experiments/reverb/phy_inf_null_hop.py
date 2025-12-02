import numpy as np
import tensorflow as tf
import soundfile as sf
import scipy.signal
import json
import os
import time

# --- 1. Load Config ---
if not os.path.exists("config.json"): 
    print("Config missing. Using defaults.")
    CONF = {
        "fs": 16000, 
        "n_fft": 1024, 
        "hop_len": 512, 
        "d": 0.04,  # 4 cm spacing
        "c": 343.0, 
        "train_seg_samples": 32000
    }
else:
    with open("config.json", "r") as f: CONF = json.load(f)

FS = CONF["fs"]
N_FFT = CONF["n_fft"]
HOP = CONF["hop_len"]
D = CONF["d"]
C_SPEED = CONF["c"] # Renamed to avoid collision with Constraint Matrix C
WIN_SIZE = CONF["train_seg_samples"]
ANGLE_TARGET = 90.0
FREQ_BINS = (N_FFT // 2) + 1
N_MICS = 2

# --- 2. Physics Helpers ---
def get_steering_vector_single(f, angle_deg, d, c):
    """Calculates theoretical steering vector for a single frequency"""
    theta = np.deg2rad(angle_deg)
    # Standard linear array geometry centered at 0
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f
    v = np.array([[np.exp(-1j * omega * tau1)], [np.exp(-1j * omega * tau2)]])
    # Phase normalize to reference mic (index 0)
    v = v / (v[0] + 1e-10)
    return v

def hybrid_hard_null_bf(Y, mask, f_bins):
    """
    Performs Hybrid Null-Steering Beamforming.
    Y: (M, F, T) - Complex STFT
    mask: (F, T) - Target Probabilities (0.0 to 1.0)
    f_bins: (F,) - Frequency values in Hz
    """
    n_freq = Y.shape[1]
    n_frames = Y.shape[2]
    S_out = np.zeros((n_freq, n_frames), dtype=complex)
    
    # Pre-allocate response vector [Target=1, Null=0]
    desired_response = np.array([[1], [0]], dtype=np.complex64)
    
    # Interference Mask = 1 - Target Mask
    mask_int = 1.0 - mask
    
    # --- Loop over Frequencies ---
    # (Looping is safer than vectorization for EVD/Solve stability checks)
    for i in range(n_freq):
        f_hz = f_bins[i]
        
        # 1. Low Frequency Bypass (< 200 Hz)
        # Spatial resolution is poor here; nulling blows up white noise.
        if f_hz < 200:
            S_out[i, :] = Y[0, i, :] # Pass Reference Mic
            continue
            
        # Extract bin data (2, T)
        Y_vec = Y[:, i, :]
        m_int_vec = mask_int[i, :] # (T,)
        
        # 2. Estimate Interference Covariance (R_int)
        # Weighted outer product: E[ Y * Y^H * mask_int ]
        # Denominator: Sum of mask weights
        denom_int = np.sum(m_int_vec) + 1e-6
        R_int = (Y_vec * m_int_vec) @ (Y_vec.conj().T) / denom_int
        
        # 3. Find Dominant Interference Direction (v_int) via EVD
        eigvals, eigvecs = np.linalg.eigh(R_int)
        v_int = eigvecs[:, -1].reshape(2, 1) # Principal Eigenvector
        
        # Normalize v_int phase relative to Mic 0
        v_int = v_int / (v_int[0] / (np.abs(v_int[0]) + 1e-10))
        
        # 4. Get Target Steering Vector (v_tgt)
        # We use the fixed geometry for the target anchor to prevent drift
        v_tgt = get_steering_vector_single(f_hz, ANGLE_TARGET, D, C_SPEED)
        
        # 5. Formulate Constraint Matrix C_mat
        # Columns: [Target Vector, Interference Vector]
        C_mat = np.column_stack((v_tgt, v_int))
        
        # 6. Safety Check: Condition Number
        # If sources are too close, C_mat becomes singular -> Weights explode
        cond_num = np.linalg.cond(C_mat)
        
        if cond_num > 10: 
            # FALLBACK: Standard Delay-and-Sum towards Target
            w = v_tgt / N_MICS
        else:
            # HARD NULL SOLUTION
            # Solve: C_mat^H * w = [1, 0]
            try:
                w = np.linalg.solve(C_mat.conj().T, desired_response)
            except np.linalg.LinAlgError:
                w = v_tgt / N_MICS
                
        # 7. Apply Spatial Filter
        # w: (2, 1), Y_vec: (2, T) -> (1, T)
        S_out[i, :] = (w.conj().T @ Y_vec).squeeze()
        
    return S_out

# --- 3. Physics-Aware TFLite Wrapper ---
class TFLiteBeamformer:
    def __init__(self, model_path="mask_estimator_phy.tflite"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_idx = self.input_details[0]['index']
        self.output_idx = self.output_details[0]['index']
        
        # Precompute Freq Map
        self.freq_map = np.linspace(0, 1, FREQ_BINS, dtype=np.float32)[:, np.newaxis]

    def predict_mask(self, log_mag, raw_ipd):
        # 1. Sin/Cos Transform
        sin_ipd = np.sin(raw_ipd)
        cos_ipd = np.cos(raw_ipd)
        
        # 2. Freq Map tiling
        t_steps = log_mag.shape[1]
        f_map_tiled = np.tile(self.freq_map, (1, t_steps))
        
        # 3. Stack (Batch, Freq, Time, 4)
        input_tensor = np.stack([log_mag, sin_ipd, cos_ipd, f_map_tiled], axis=-1)
        input_tensor = input_tensor[np.newaxis, ...].astype(np.float32)
        
        # 4. Resize if needed
        current_shape = self.input_details[0]['shape']
        if not np.array_equal(input_tensor.shape, current_shape):
            self.interpreter.resize_tensor_input(self.input_idx, input_tensor.shape)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(self.input_idx, input_tensor)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_idx)
        return output_data.squeeze() 

# --- 4. Pipeline ---
def process_audio_file(input_path, output_path, model_path="mask_estimator_phy.tflite"):
    print(f"Processing {input_path}...")
    y, sr = sf.read(input_path, dtype='float32')
    if sr != FS: print(f"Warning: SR mismatch. Input: {sr}, Config: {FS}")
    
    # Handle Mono Input
    if y.ndim == 1:
        print("Error: Input is mono. Requires 2 channels.")
        return

    chunk_size = WIN_SIZE
    hop_size = chunk_size // 2
    out_buf = np.zeros(len(y))
    norm_buf = np.zeros(len(y))
    
    # Initialize Model
    try:
        bf = TFLiteBeamformer(model_path)
    except Exception as e:
        print(f"Failed to load TFLite model: {e}")
        return
    
    num_chunks = int(np.ceil(len(y) / hop_size))
    
    start_time = time.time()
    
    for i in range(num_chunks):
        start = i * hop_size
        end = start + chunk_size
        chunk = y[start:end]
        
        # Padding
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, ((0, chunk_size - len(chunk)), (0, 0)))
            
        # STFT
        f_bins, t_bins, Y = scipy.signal.stft(chunk.T, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)
        # Y shape: (M, F, T)
        
        # Feature Prep
        mag = np.abs(Y)
        log_mag = np.log(mag[0] + 1e-7)
        ipd = np.angle(Y[0]) - np.angle(Y[1])
        
        # 1. Inference (Get Mask)
        mask = bf.predict_mask(log_mag, ipd)
        
        # Cropping to match valid STFT frames (TFLite sometimes pads output)
        min_f, min_t = mask.shape
        Y = Y[:, :min_f, :min_t]
        f_valid = f_bins[:min_f]
        mask = mask[:min_f, :min_t]
        
        # 2. Hybrid Hard-Null Beamforming
        # (Replaces batch_mvdr)
        S_out = hybrid_hard_null_bf(Y, mask, f_valid)
        
        # 3. Post-Filtering (Spectral)
        # Apply Soft Mask to remove residual diffuse noise
        S_final = S_out * mask 
        
        # ISTFT
        _, chunk_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)
        
        # OLA
        w_len = min(len(chunk_out), len(out_buf[start:]))
        out_buf[start:start+w_len] += chunk_out[:w_len]
        norm_buf[start:start+w_len] += 1.0
        
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.3f}s")
        
    # Finalize
    final = out_buf / np.maximum(norm_buf, 1.0)
    # Peak Norm
    final = final / (np.max(np.abs(final)) + 1e-9)
    
    sf.write(output_path, final, FS)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    if os.path.exists("sample/mixture.wav") and os.path.exists("mask_estimator_phy.tflite"):
        process_audio_file("sample/mixture.wav", "enhanced_physics_hybrid.wav")
    else:
        print("Ensure 'mixture_TEST.wav' and 'mask_estimator_phy.tflite' exist.")