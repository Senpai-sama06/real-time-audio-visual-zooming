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

    CONF = {"fs": 16000, "n_fft": 1024, "hop_len": 512, "d": 0.04, "c": 343.0, "train_seg_samples": 32000}

else:

    with open("config.json", "r") as f: CONF = json.load(f)



FS = CONF["fs"]

N_FFT = CONF["n_fft"]

HOP = CONF["hop_len"]

D = CONF["d"]

C = CONF["c"]

WIN_SIZE = CONF["train_seg_samples"]

ANGLE_TARGET = 90.0

SIGMA = 1e-5

N_MICS = 2



# --- 2. Vectorized Physics Helpers (The Speedup) ---

def get_all_steering_vectors(f_bins, angle_deg, d, c):

    """

    Pre-computes vectors for ALL valid frequencies at once.

    Returns: (Freqs, Mics, 1)

    """

    theta = np.deg2rad(angle_deg)

    tau1 = (d / 2) * np.cos(theta) / c

    tau2 = (d / 2) * np.cos(theta - np.pi) / c

    

    omega = 2 * np.pi * f_bins

    # Stack phases: (2, Freqs)

    sv = np.stack([np.exp(-1j * omega * tau1), np.exp(-1j * omega * tau2)], axis=0)

    

    # Transpose to (Freqs, Mics, 1) for broadcasting

    return np.expand_dims(sv.T, axis=-1)



def batch_mvdr(Y, mask, f_bins, d_vectors, sigma):

    """

    Computes MVDR for all frequencies simultaneously using matrix broadcasting.

    This replaces the slow 'for i in range(513)' loop.

    """

    # Y shape: (Mics, Freqs, Time) -> Permute to (Freqs, Mics, Time)

    Y_perm = np.transpose(Y, (1, 0, 2))

    

    # Noise Mask: (Freqs, 1, Time)

    mask_noise = (1.0 - mask)[:, np.newaxis, :]

    

    # Calculate Spatial Covariance Matrix (R)

    # Weighted signal: Y_w = Y * sqrt(mask)

    Y_w = Y_perm * np.sqrt(mask_noise + 1e-10)

    

    # R = Y_w @ Y_w^H (Summed over Time dimension)

    # einsum 'fmt,fnt->fmn' -> Freq, Mic, Mic

    R = np.einsum('fmt,fnt->fmn', Y_w, Y_w.conj())

    

    # Normalize by sum of mask weights

    norm = np.sum(mask_noise, axis=2)[:, :, np.newaxis] + 1e-6

    R = R / norm

    

    # Diagonal Loading (Regularization)

    # Add sigma identity matrix to every frequency bin

    eye = np.eye(N_MICS)[np.newaxis, :, :]

    R += sigma * eye

    

    # Solve MVDR: R * w = d

    # np.linalg.solve broadcasts over the first dimension (Freqs)

    try:

        w_unnorm = np.linalg.solve(R, d_vectors)

    except np.linalg.LinAlgError:

        # Fallback

        w_unnorm = np.zeros_like(d_vectors)

        w_unnorm[:, 0, :] = 1.0



    # Normalize weights

    # w = w_unnorm / (d^H * w_unnorm)

    d_H = np.transpose(d_vectors.conj(), (0, 2, 1))

    denom = np.matmul(d_H, w_unnorm) + 1e-10

    w = w_unnorm / denom

    

    # Apply Beamforming: S = w^H * Y

    w_H = np.transpose(w.conj(), (0, 2, 1))

    S_out = np.matmul(w_H, Y_perm)

    

    return S_out.squeeze(1) # (Freqs, Time)



# --- 3. TFLite Wrapper ---

class TFLiteBeamformer:

    def __init__(self, model_path="mask_estimator.tflite"):

        self.interpreter = tf.lite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()

        

        self.input_details = self.interpreter.get_input_details()

        self.output_details = self.interpreter.get_output_details()

        self.input_idx = self.input_details[0]['index']

        self.output_idx = self.output_details[0]['index']

        

    def predict_mask(self, log_mag, ipd):

        # Prepare Input (Batch, Freq, Time, Chan)

        input_tensor = np.stack([log_mag, ipd], axis=-1).astype(np.float32)

        input_tensor = input_tensor[np.newaxis, ...]

        

        # Dynamic Resizing

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



# --- 4. Processing Pipeline ---

def process_audio_file(input_path, output_path, model_path="mask_estimator.tflite"):

    # A. Stats

    size_mb = os.path.getsize(model_path) / 1024 / 1024

    print(f"Model Size:       {size_mb:.2f} MB")

    

    # B. Load

    y, sr = sf.read(input_path, dtype='float32')

    if sr != FS: print("Warning: SR mismatch")

    print(f"Audio Duration:   {len(y)/FS:.2f}s")

    

    # C. Prep Chunks

    chunk_size = WIN_SIZE

    hop_size = chunk_size // 2

    out_buf = np.zeros(len(y))

    norm_buf = np.zeros(len(y))

    

    bf = TFLiteBeamformer(model_path)

    

    start_time = time.time()

    num_chunks = int(np.ceil(len(y) / hop_size))

    

    for i in range(num_chunks):

        start = i * hop_size

        end = start + chunk_size

        

        chunk = y[start:end]

        if len(chunk) < chunk_size:

            chunk = np.pad(chunk, ((0, chunk_size - len(chunk)), (0, 0)))

            

        # 1. STFT

        # Transpose chunk to (Mics, Time) -> STFT (Mics, Freqs, Time)

        f_bins, t_bins, Y = scipy.signal.stft(chunk.T, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)

        

        # 2. Inference

        mag = np.abs(Y)

        log_mag = np.log(mag[0] + 1e-7)

        ipd = np.angle(Y[0]) - np.angle(Y[1])

        

        mask = bf.predict_mask(log_mag, ipd)

        

        # Align Shapes

        min_f = min(mask.shape[0], Y.shape[1])

        min_t = min(mask.shape[1], Y.shape[2])

        mask = mask[:min_f, :min_t]

        Y = Y[:, :min_f, :min_t]

        f_valid = f_bins[:min_f]

        

        # 3. Vectorized MVDR (No Loops!)

        d_vecs = get_all_steering_vectors(f_valid, ANGLE_TARGET, D, C)

        S_out = batch_mvdr(Y, mask, f_valid, d_vecs, SIGMA)

        

        # 4. ISTFT

        S_final = S_out * np.maximum(mask, 0.05)

        _, chunk_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)

        

        # 5. Overlap Add

        w_len = min(len(chunk_out), len(out_buf[start:]))

        out_buf[start:start+w_len] += chunk_out[:w_len]

        norm_buf[start:start+w_len] += 1.0

        

    proc_time = time.time() - start_time

    

    # Normalize

    norm_buf[norm_buf == 0] = 1.0

    final = out_buf / norm_buf

    final = final / (np.max(np.abs(final)) + 1e-9)

    

    sf.write(output_path, final, FS)

    

    print("-" * 40)

    print(f"Total Inference Time: {proc_time:.4f}s")

    print(f"Real-Time Factor:     {proc_time / (len(y)/FS):.4f}x")

    print(f"Saved to:             {output_path}")

    print("-" * 40)



if __name__ == "__main__":

    input_file = "mixture_TEST.wav"

    if os.path.exists(input_file) and os.path.exists("mask_estimator.tflite"):

        process_audio_file(input_file, "enhanced_tflite_opt.wav")

    else:

        print("Run world_building.py and tf_model.py first.")