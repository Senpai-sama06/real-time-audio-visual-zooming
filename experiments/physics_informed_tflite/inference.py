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
FREQ_BINS = (N_FFT // 2) + 1
N_MICS = 2

# --- 2. Vectorized Physics Helpers ---
def get_all_steering_vectors(f_bins, angle_deg, d, c):
    theta = np.deg2rad(angle_deg)
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f_bins
    sv = np.stack([np.exp(-1j * omega * tau1), np.exp(-1j * omega * tau2)], axis=0)
    return np.expand_dims(sv.T, axis=-1)

def batch_mvdr(Y, mask, f_bins, d_vectors, sigma=1e-5):
    Y_perm = np.transpose(Y, (1, 0, 2))
    mask_noise = (1.0 - mask)[:, np.newaxis, :]
    
    Y_w = Y_perm * np.sqrt(mask_noise + 1e-10)
    R = np.einsum('fmt,fnt->fmn', Y_w, Y_w.conj())
    
    norm = np.sum(mask_noise, axis=2)[:, :, np.newaxis] + 1e-6
    R = R / norm
    
    eye = np.eye(N_MICS)[np.newaxis, :, :]
    R += sigma * eye
    
    try:
        w_unnorm = np.linalg.solve(R, d_vectors)
    except np.linalg.LinAlgError:
        w_unnorm = np.zeros_like(d_vectors)
        w_unnorm[:, 0, :] = 1.0

    d_H = np.transpose(d_vectors.conj(), (0, 2, 1))
    denom = np.matmul(d_H, w_unnorm) + 1e-10
    w = w_unnorm / denom
    
    w_H = np.transpose(w.conj(), (0, 2, 1))
    S_out = np.matmul(w_H, Y_perm)
    return S_out.squeeze(1)

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
        # 1. Sin/Cos Transfrom
        sin_ipd = np.sin(raw_ipd)
        cos_ipd = np.cos(raw_ipd)
        
        # 2. Freq Map tiling
        t_steps = log_mag.shape[1]
        f_map_tiled = np.tile(self.freq_map, (1, t_steps))
        
        # 3. Stack (Batch, Freq, Time, 4)
        input_tensor = np.stack([log_mag, sin_ipd, cos_ipd, f_map_tiled], axis=-1)
        input_tensor = input_tensor[np.newaxis, ...].astype(np.float32)
        
        # 4. Resize if needed
        # We need to ensure the Frequency dimension (dim 1) matches training
        # But Time dimension (dim 2) can vary
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
    y, sr = sf.read(input_path, dtype='float32')
    if sr != FS: print("Warning: SR mismatch")
    
    chunk_size = WIN_SIZE
    hop_size = chunk_size // 2
    out_buf = np.zeros(len(y))
    norm_buf = np.zeros(len(y))
    
    bf = TFLiteBeamformer(model_path)
    
    num_chunks = int(np.ceil(len(y) / hop_size))
    
    for i in range(num_chunks):
        start = i * hop_size
        end = start + chunk_size
        chunk = y[start:end]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, ((0, chunk_size - len(chunk)), (0, 0)))
            
        f_bins, t_bins, Y = scipy.signal.stft(chunk.T, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)
        
        # Feature Prep
        mag = np.abs(Y)
        log_mag = np.log(mag[0] + 1e-7)
        ipd = np.angle(Y[0]) - np.angle(Y[1])
        
        # Inference
        mask = bf.predict_mask(log_mag, ipd)
        
        # MVDR
        min_f, min_t = mask.shape
        Y = Y[:, :min_f, :min_t]
        f_valid = f_bins[:min_f]
        
        d_vecs = get_all_steering_vectors(f_valid, ANGLE_TARGET, D, C)
        S_out = batch_mvdr(Y, mask, f_valid, d_vecs)
        
        # Post-Filtering
        S_final = S_out * mask 
        
        _, chunk_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)
        
        w_len = min(len(chunk_out), len(out_buf[start:]))
        out_buf[start:start+w_len] += chunk_out[:w_len]
        norm_buf[start:start+w_len] += 1.0
        
    final = out_buf / np.maximum(norm_buf, 1.0)
    final = final / (np.max(np.abs(final)) + 1e-9)
    sf.write(output_path, final, FS)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    start=time.time()
    if os.path.exists("mixture_TEST.wav") and os.path.exists("mask_estimator_phy.tflite"):
        process_audio_file("mixture_TEST.wav", "enhanced_physics.wav")
        end=time.time()
        print(f"inference time: {end-start}")
    else:
        print("Run world_building.py then model_training.py")



'''
for 2 seconds  ----->   Inference (avg): 0.699438 seconds

C:\Users\DHRUV SINGH\Downloads\platform-tools-latest-windows\platform-tools>adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mask_estimator_phy.tflite --num_threads=4
INFO: STARTING!
INFO: Log parameter values verbosely: [0]
INFO: Num threads: [4]
INFO: Graph: [/data/local/tmp/mask_estimator_phy.tflite]
INFO: Signature to run: []
INFO: #threads used for CPU inference: [4]
INFO: Loaded model /data/local/tmp/mask_estimator_phy.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
VERBOSE: Replacing 49 out of 49 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions for subgraph 0.
INFO: The input model file size (MB): 16.1527
INFO: Initialized session in 102.32ms.
INFO: Running benchmark for at least 1 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.
INFO: count=1 curr=670083 p5=670083 median=670083 p95=670083

INFO: Running benchmark for at least 50 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
INFO: count=50 first=591245 curr=917186 min=502355 max=2270688 avg=699438 std=325425 p5=502674 median=576856 p95=1373207

INFO: Inference timings in us: Init: 102320, First inference: 670083, Warmup (avg): 670083, Inference (avg): 699438
INFO: Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
INFO: Memory footprint delta from the start of the tool (MB): init=98.0977 overall=99.2031

'''        


'''
if used gpu----->   Inference (avg): 0.309350 seconds

sk_estimator_phy.tflite --num_threads=2 --use_gpu=true --use_nnapi=true
INFO: STARTING!
INFO: Log parameter values verbosely: [0]
INFO: Num threads: [2]
INFO: Graph: [/data/local/tmp/mask_estimator_phy.tflite]
INFO: Signature to run: []
INFO: #threads used for CPU inference: [2]
INFO: Use gpu: [1]
INFO: Use NNAPI: [1]
INFO: NNAPI accelerators available: [nnapi-reference]
INFO: Loaded model /data/local/tmp/mask_estimator_phy.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite delegate for GPU.
INFO: GPU delegate created.
INFO: Created TensorFlow Lite delegate for NNAPI.
INFO: Going to apply 2 delegates one after another.
INFO: Loaded OpenCL library with dlopen.
VERBOSE: Replacing 49 out of 49 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 1 partitions for subgraph 0.
INFO: Initialized OpenCL-based API.
INFO: Created 1 GPU delegate kernels.
INFO: Explicitly applied GPU delegate, and the model graph will be completely executed by the delegate.
INFO: Though NNAPI delegate is explicitly applied, the model graph will not be executed by the delegate.
INFO: The input model file size (MB): 16.1527
INFO: Initialized session in 5664.36ms.
INFO: Running benchmark for at least 1 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.
INFO: count=2 first=328140 curr=309932 min=309932 max=328140 avg=319036 std=9104 p5=309932 median=328140 p95=328140

INFO: Running benchmark for at least 50 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
INFO: count=50 first=307215 curr=309783 min=305938 max=333030 avg=309350 std=4411 p5=306372 median=307968 p95=315711

INFO: Inference timings in us: Init: 5664363, First inference: 328140, Warmup (avg): 319036, Inference (avg): 309350
INFO: Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
INFO: Memory footprint delta from the start of the tool (MB): init=375.48 overall=375.48

'''