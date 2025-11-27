import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np
import os
import json
import glob
import scipy.signal
import librosa
import random

# --- 1. Load Config ---
if not os.path.exists("config.json"): raise FileNotFoundError("Run world_building.py first")
with open("config.json", "r") as f: CONF = json.load(f)

# Constants
FREQ_BINS = 513  # (N_FFT / 2) + 1
INPUT_CHANNELS = 2 # Log-Magnitude + IPD

# Training Constants
BATCH_SIZE = 4
STEPS_PER_EPOCH = 10  # Small number for demo speed
EPOCHS = 40

# --- 2. Data Generator (On-the-Fly Mixing) ---
class AudioGenerator(tf.keras.utils.Sequence):
    """Generates batches of spatial audio features on the fly."""
    def __init__(self, wav_files, batch_size=4):
        self.wav_files = wav_files
        self.batch_size = batch_size
        self.fs = CONF["fs"]
        self.seg_len = CONF["train_seg_samples"]
        self.n_fft = CONF["n_fft"]
        self.hop = CONF["hop_len"]
        self.d = CONF["d"]
        self.c = CONF["c"]
        
    def __len__(self):
        return STEPS_PER_EPOCH

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
            # Pick 3 random files (Target, Int1, Int2)
            samples = random.sample(self.wav_files, 3)
            raws = []
            for s in samples:
                y, _ = librosa.load(s, sr=self.fs)
                if len(y) < self.seg_len: y = np.pad(y, (0, self.seg_len - len(y)))
                start = random.randint(0, len(y) - self.seg_len)
                raws.append(y[start:start+self.seg_len])

            # Mix
            m1, m2, tgt, int_sig = np.zeros(self.seg_len), np.zeros(self.seg_len), np.zeros(self.seg_len), np.zeros(self.seg_len)
            
            configs = [(90, 'tgt'), (40, 'int'), (130, 'int')]
            
            for i, (angle, role) in enumerate(configs):
                t1, t2 = self._calc_delay(angle)
                s1 = self._apply_delay(raws[i], t1)
                m1 += s1
                m2 += self._apply_delay(raws[i], t2)
                if role == 'tgt': tgt += s1
                else: int_sig += s1

            # Feature Extraction
            f, t, Z = scipy.signal.stft(np.stack([m1, m2]), fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
            _, _, S_t = scipy.signal.stft(tgt, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
            _, _, S_i = scipy.signal.stft(int_sig, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)

            # Inputs: (Freq, Time, 2)
            mag = np.abs(Z)
            log_mag = np.log(mag[0] + 1e-7)
            ipd = np.angle(Z[0]) - np.angle(Z[1])
            feat = np.stack([log_mag, ipd], axis=-1) # NHWC format for TF
            
            # Target Mask: (Freq, Time, 1)
            mask = np.where(np.abs(S_t) > np.abs(S_i), 1.0, 0.0).astype(np.float32)
            mask = mask[..., np.newaxis]

            X_batch.append(feat)
            Y_batch.append(mask)

        return np.array(X_batch), np.array(Y_batch)

# --- 3. Model Definition (DeepFPU Keras) ---
def res_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_model():
    inputs = Input(shape=(FREQ_BINS, None, INPUT_CHANNELS))
    
    # Encoder
    e1 = layers.Conv2D(32, 3, padding='same')(inputs)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.ReLU()(e1)
    e1 = layers.Conv2D(32, 3, padding='same')(e1)
    e1 = layers.BatchNormalization()(e1)
    e1 = layers.ReLU()(e1)
    
    # Pool only Time axis (1, 2)
    p1 = layers.MaxPooling2D((1, 2))(e1) 
    
    e2 = layers.Conv2D(64, 3, padding='same')(p1)
    e2 = res_block(e2, 64)
    p2 = layers.MaxPooling2D((1, 2))(e2)
    
    e3 = layers.Conv2D(128, 3, padding='same')(p2)
    e3 = res_block(e3, 128)
    p3 = layers.MaxPooling2D((1, 2))(e3)
    
    # Bottleneck
    b = layers.Conv2D(256, 3, padding='same')(p3)
    b = res_block(b, 256)
    
    # Decoder
    u3 = layers.Conv2DTranspose(128, (1, 2), strides=(1, 2), padding='same')(b)
    c3 = layers.Concatenate(axis=-1)([u3, e3])
    d3 = layers.Conv2D(128, 3, padding='same')(c3)
    d3 = res_block(d3, 128)
    
    u2 = layers.Conv2DTranspose(64, (1, 2), strides=(1, 2), padding='same')(d3)
    c2 = layers.Concatenate(axis=-1)([u2, e2])
    d2 = layers.Conv2D(64, 3, padding='same')(c2)
    d2 = res_block(d2, 64)
    
    u1 = layers.Conv2DTranspose(32, (1, 2), strides=(1, 2), padding='same')(d2)
    c1 = layers.Concatenate(axis=-1)([u1, e1])
    d1 = layers.Conv2D(32, 3, padding='same')(c1)
    d1 = res_block(d1, 32)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d1)
    
    return models.Model(inputs, outputs)

# --- 4. Main Execution ---
def main():
    # 1. Locate Files
    try:
        # Try finding files downloaded by world_building.py
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
    except:
        # Fallback
        home = os.path.expanduser("~")
        wav_path = os.path.join(home, ".cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset/versions/1/LJSpeech-1.1/wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
        
    if len(files) < 10:
        print("Data not found. Please run world_building.py first.")
        return

    # 2. Train
    print("[TF] Building and Training Keras Model...")
    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    gen = AudioGenerator(files, batch_size=BATCH_SIZE)
    model.fit(gen, epochs=EPOCHS)
    
    # 3. Convert to TFLite with Quantization
    print("[TF] Converting to TFLite (Dynamic Quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open("mask_estimator.tflite", "wb") as f:
        f.write(tflite_model)
        
    print("[TF] Saved 'mask_estimator.tflite' successfully.")

if __name__ == "__main__":
    main()