import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import os
import json
import glob
import scipy.signal
import librosa
import random

# --- 0. GPU Configuration (New) ---
# Ensure TensorFlow uses the GPU and allows memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[TF] GPUs Detected: {len(gpus)}")
        print(f"[TF] Using GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)
else:
    print("[TF] No GPU detected. Training will be slow on CPU.")

# --- 1. Load Config ---
if not os.path.exists("config.json"): raise FileNotFoundError("Run world_building.py first")
with open("config.json", "r") as f: CONF = json.load(f)

FREQ_BINS = (CONF["n_fft"] // 2) + 1
INPUT_CHANNELS = 4 

# --- NPU REQUIREMENT: Fixed Time Dimension ---
# 2.0 seconds audio -> 64 frames (approx)
FIXED_TIME_STEPS = 64 

BATCH_SIZE = 4
STEPS_PER_EPOCH = 50 
EPOCHS = 60

# --- 2. Physics-Informed Data Generator (Standard) ---
class PhysicsAudioGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_files, batch_size=4):
        self.wav_files = wav_files
        self.batch_size = batch_size
        self.fs = CONF["fs"]
        self.seg_len = CONF["train_seg_samples"]
        self.n_fft = CONF["n_fft"]
        self.hop = CONF["hop_len"]
        self.d = CONF["d"]
        self.c = CONF["c"]
        self.freq_map = np.linspace(0, 1, FREQ_BINS, dtype=np.float32)[:, np.newaxis]

    def __len__(self): return STEPS_PER_EPOCH

    def _calc_delay(self, angle):
        theta = np.deg2rad(angle)
        return (self.d/2)*np.cos(theta)/self.c, (self.d/2)*np.cos(theta-np.pi)/self.c

    def _apply_delay(self, y, delay):
        n = len(y)
        y_fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n, 1.0/self.fs)
        return np.fft.irfft(y_fft * np.exp(-1j*2*np.pi*freqs*delay), n=n)

    def __getitem__(self, index):
        X_batch, Y_batch = [], []
        for _ in range(self.batch_size):
            samples = random.sample(self.wav_files, 3)
            raws = []
            for s in samples:
                y, _ = librosa.load(s, sr=self.fs)
                if len(y) < self.seg_len: y = np.pad(y, (0, self.seg_len - len(y)))
                start = random.randint(0, len(y) - self.seg_len)
                raws.append(y[start:start+self.seg_len])

            m1, m2 = np.zeros(self.seg_len), np.zeros(self.seg_len)
            tgt_sig, int_sig = np.zeros(self.seg_len), np.zeros(self.seg_len)
            configs = [(90, 'tgt'), (40, 'int'), (130, 'int')]
            
            for i, (angle, role) in enumerate(configs):
                t1, t2 = self._calc_delay(angle)
                s1 = self._apply_delay(raws[i], t1)
                m1 += s1; m2 += self._apply_delay(raws[i], t2)
                if role == 'tgt': tgt_sig += s1
                else: int_sig += s1

            _, _, Z = scipy.signal.stft(np.stack([m1, m2]), fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
            _, _, S_t = scipy.signal.stft(tgt_sig, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
            _, _, S_i = scipy.signal.stft(int_sig, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)

            mag = np.abs(Z); log_mag = np.log(mag[0] + 1e-7)
            raw_ipd = np.angle(Z[0]) - np.angle(Z[1])
            sin_ipd = np.sin(raw_ipd); cos_ipd = np.cos(raw_ipd)
            
            # --- Padding to Fixed Time Steps (NPU Critical) ---
            curr_t = log_mag.shape[1]
            if curr_t < FIXED_TIME_STEPS:
                pad = FIXED_TIME_STEPS - curr_t
                log_mag = np.pad(log_mag, ((0,0),(0,pad)))
                sin_ipd = np.pad(sin_ipd, ((0,0),(0,pad)))
                cos_ipd = np.pad(cos_ipd, ((0,0),(0,pad)))
                t_steps = FIXED_TIME_STEPS
            else:
                log_mag = log_mag[:, :FIXED_TIME_STEPS]
                sin_ipd = sin_ipd[:, :FIXED_TIME_STEPS]
                cos_ipd = cos_ipd[:, :FIXED_TIME_STEPS]
                t_steps = FIXED_TIME_STEPS

            f_map_tiled = np.tile(self.freq_map, (1, t_steps))
            feat = np.stack([log_mag, sin_ipd, cos_ipd, f_map_tiled], axis=-1)

            mag_t = np.abs(S_t); mag_i = np.abs(S_i)
            if mag_t.shape[1] < FIXED_TIME_STEPS:
                pad = FIXED_TIME_STEPS - mag_t.shape[1]
                mag_t = np.pad(mag_t, ((0,0),(0,pad)))
                mag_i = np.pad(mag_i, ((0,0),(0,pad)))
            else:
                mag_t = mag_t[:, :FIXED_TIME_STEPS]
                mag_i = mag_i[:, :FIXED_TIME_STEPS]

            irm = mag_t / (mag_t + mag_i + 1e-7)
            irm = irm[..., np.newaxis] 
            X_batch.append(feat); Y_batch.append(irm)

        return np.array(X_batch), np.array(Y_batch)

# --- 3. Loss (Unchanged) ---
class PhysicsWeightedLoss(tf.keras.losses.Loss):
    def __init__(self, fs, n_fft, cutoff_hz=500, w_low=0.1, w_high=1.0, **kwargs):
        super().__init__(**kwargs)
        self.fs = fs; self.n_fft = n_fft; self.cutoff_hz = cutoff_hz
        self.w_low = w_low; self.w_high = w_high
        freqs = np.fft.rfftfreq(n_fft, 1/fs)
        weights = np.where(freqs < cutoff_hz, w_low, w_high)
        self.W = tf.constant(weights.reshape(1, -1, 1, 1), dtype=tf.float32)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred) * self.W)

# --- 4. Deep Architecture (Replicating PyTorch Complexity) ---
def res_block(x, filters):
    shortcut = x
    # Conv 1
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Conv 2
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_complex_unet():
    # NPU Safe Input
    inputs = Input(shape=(FREQ_BINS, FIXED_TIME_STEPS, INPUT_CHANNELS))
    
    # --- Encoder 1 (32 Filters) ---
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    e1 = layers.Conv2D(32, 3, padding='same')(x)
    e1 = layers.BatchNormalization()(e1); e1 = layers.ReLU()(e1)
    
    p1 = layers.MaxPooling2D((1, 2))(e1) 

    # --- Encoder 2 (64 Filters) ---
    x = layers.Conv2D(64, 3, padding='same')(p1)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    e2 = res_block(x, 64)
    
    p2 = layers.MaxPooling2D((1, 2))(e2)

    # --- Encoder 3 (128 Filters) ---
    x = layers.Conv2D(128, 3, padding='same')(p2)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    e3 = res_block(x, 128)
    
    p3 = layers.MaxPooling2D((1, 2))(e3)

    # --- Encoder 4 (256 Filters) --- 
    # This layer was missing in the previous "simple" version
    x = layers.Conv2D(256, 3, padding='same')(p3)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    e4 = res_block(x, 256)
    
    p4 = layers.MaxPooling2D((1, 2))(e4)

    # --- Bottleneck (512 Filters) ---
    x = layers.Conv2D(512, 3, padding='same')(p4)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = res_block(x, 512)
    b = res_block(x, 512) # Double ResBlock deep in network

    # --- Decoder 4 (256 Filters) ---
    u4 = layers.Conv2DTranspose(256, (1, 2), strides=(1, 2), padding='same')(b)
    # Resizing ensures NPU doesn't crash on slight pixel mismatches
    if u4.shape[2] != e4.shape[2]: u4 = layers.Resizing(e4.shape[1], e4.shape[2])(u4)
    c4 = layers.Concatenate()([u4, e4])
    
    x = layers.Conv2D(256, 3, padding='same')(c4)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    d4 = res_block(x, 256)

    # --- Decoder 3 (128 Filters) ---
    u3 = layers.Conv2DTranspose(128, (1, 2), strides=(1, 2), padding='same')(d4)
    if u3.shape[2] != e3.shape[2]: u3 = layers.Resizing(e3.shape[1], e3.shape[2])(u3)
    c3 = layers.Concatenate()([u3, e3])
    
    x = layers.Conv2D(128, 3, padding='same')(c3)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    d3 = res_block(x, 128)

    # --- Decoder 2 (64 Filters) ---
    u2 = layers.Conv2DTranspose(64, (1, 2), strides=(1, 2), padding='same')(d3)
    if u2.shape[2] != e2.shape[2]: u2 = layers.Resizing(e2.shape[1], e2.shape[2])(u2)
    c2 = layers.Concatenate()([u2, e2])
    
    x = layers.Conv2D(64, 3, padding='same')(c2)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    d2 = res_block(x, 64)

    # --- Decoder 1 (32 Filters) ---
    u1 = layers.Conv2DTranspose(32, (1, 2), strides=(1, 2), padding='same')(d2)
    if u1.shape[2] != e1.shape[2]: u1 = layers.Resizing(e1.shape[1], e1.shape[2])(u1)
    c1 = layers.Concatenate()([u1, e1])
    
    x = layers.Conv2D(32, 3, padding='same')(c1)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    d1 = layers.Conv2D(32, 3, padding='same')(x)
    d1 = layers.BatchNormalization()(d1); d1 = layers.ReLU()(d1)

    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d1)
    return models.Model(inputs, outputs)

# --- 5. Main ---
def main():
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
    except:
        home = os.path.expanduser("~")
        wav_path = os.path.join(home, ".cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset/versions/1/LJSpeech-1.1/wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))

    if len(files) < 10: return

    print("[TF] Building Complex NPU-Ready DeepFPU (Fixed 64 Frames)...")
    # Using 'with tf.device' is usually optional as Keras does it automatically,
    # but this explicit scope ensures model creation happens on GPU.
    with tf.device('/GPU:0'):
        model = build_complex_unet()
        model.summary() 
        
        loss_fn = PhysicsWeightedLoss(
            fs=CONF["fs"], 
            n_fft=CONF["n_fft"],
            cutoff_hz=CONF["freq_cutoff_hz"],
            w_low=CONF["low_freq_weight"],
            w_high=CONF["high_freq_weight"]
        )
        
        model.compile(optimizer='adam', loss=loss_fn)
        
        gen = PhysicsAudioGenerator(files, batch_size=BATCH_SIZE)
        model.fit(gen, epochs=EPOCHS)
    
    print("[TF] Exporting Frozen Graph for NPU...")
    
    # --- CONCRETE FUNCTION EXPORT (LOCKS SHAPE) ---
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, FREQ_BINS, FIXED_TIME_STEPS, INPUT_CHANNELS], model.inputs[0].dtype)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open("mask_estimator_phy.tflite", "wb") as f:
        f.write(tflite_model)
    print("Done. Saved 'mask_estimator_phy.tflite'.")

if __name__ == "__main__":
    main()