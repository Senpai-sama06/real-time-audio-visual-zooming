import numpy as np
import soundfile as sf
import os
import kagglehub
import glob
import random
import librosa
import json

# --- 1. Global Constants ---
FS = 16000          
C = 343.0           
D = 0.04            
ANGLE_TARGET = 90.0
ANGLE_INTERFERER_A = 40.0
ANGLE_INTERFERER_B = 130.0

# STFT Parameters (Fixed for model input consistency)
N_FFT = 1024
HOP_LEN = 512
# We define a fixed training segment length (2 seconds)
TRAIN_SEG_SAMPLES = int(2.0 * FS) 

# --- 2. Configuration Saver ---
def save_config():
    config = {
        "fs": FS,
        "n_fft": N_FFT,
        "hop_len": HOP_LEN,
        "d": D,
        "c": C,
        "train_seg_samples": TRAIN_SEG_SAMPLES
    }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("[WorldBuilding] Saved config.json")

# --- 3. Audio Physics Helpers ---
def calculate_far_field_delays(azimuth_deg, d, c):
    theta_rad = np.deg2rad(azimuth_deg)
    # Mic 1 (Reference) vs Mic 2 delay calculation
    tau_m1 = (d / 2) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(theta_rad - np.pi) / c
    return tau_m1, tau_m2

def apply_frac_delay(y, delay_sec, fs):
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    # Phase shift theorem for time delay
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    y_delayed = np.fft.irfft(y_fft * phase_shift, n=n)
    return y_delayed

def load_resample(path):
    # Load as mono
    y, sr = sf.read(path, dtype='float32')
    if len(y.shape) > 1: y = np.mean(y, axis=1)
    if sr != FS: y = librosa.resample(y, orig_sr=sr, target_sr=FS)
    return y

# --- 4. Mixing Logic ---
def mix_and_save(files, prefix):
    if len(files) < 3:
        print(f"Not enough files to generate {prefix} sample.")
        return

    sources = [
        {'f': files[0], 'a': ANGLE_TARGET, 'r': 'Target'},
        {'f': files[1], 'a': ANGLE_INTERFERER_A, 'r': 'Int'},
        {'f': files[2], 'a': ANGLE_INTERFERER_B, 'r': 'Int'}
    ]
    
    raws = [load_resample(s['f']) for s in sources]
    max_l = max(len(r) for r in raws)
    
    # Pad all sources to the maximum length
    raws = [np.pad(r, (0, max_l - len(r))) for r in raws]

    m1 = np.zeros(max_l)
    m2 = np.zeros(max_l)
    tgt_ref = np.zeros(max_l)
    int_ref = np.zeros(max_l)

    for i, s in enumerate(sources):
        t1, t2 = calculate_far_field_delays(s['a'], D, C)
        
        # Simulate propagation to Mic 1 and Mic 2
        s_m1 = apply_frac_delay(raws[i], t1, FS)
        s_m2 = apply_frac_delay(raws[i], t2, FS)
        
        m1 += s_m1
        m2 += s_m2
        
        # Accumulate reference signals (at Mic 1)
        if s['r'] == 'Target': 
            tgt_ref += s_m1
        else: 
            int_ref += s_m1

    # Stack into stereo mixture
    mix = np.stack([m1, m2], axis=1)
    
    # Normalize to prevent clipping
    norm = np.max(np.abs(mix)) + 1e-9
    
    sf.write(f"mixture_{prefix}.wav", mix / norm, FS)
    sf.write(f"target_ref_{prefix}.wav", tgt_ref / norm, FS)
    sf.write(f"interf_ref_{prefix}.wav", int_ref / norm, FS)
    print(f"[WorldBuilding] Created mixture_{prefix}.wav")

def main():
    save_config()
    
    try:
        print("[WorldBuilding] Downloading/Locating dataset via KaggleHub...")
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
    except Exception as e:
        # Fallback for local Kaggle environment
        home = os.path.expanduser("~")
        wav_path = os.path.join(home, ".cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset/versions/1/LJSpeech-1.1/wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
        
    if not files:
        print("Error: Data not found. Please check your internet connection or Kaggle dataset path.")
        return

    random.shuffle(files)
    
    # Create one Training file and one Testing file
    mix_and_save(files[:3], "TRAIN")
    mix_and_save(files[3:6], "TEST")

if __name__ == "__main__":
    main()