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

# STFT Parameters
N_FFT = 1024
HOP_LEN = 512
TRAIN_SEG_SAMPLES = int(2.0 * FS) 

# --- 2. Configuration Saver ---
def save_config():
    config = {
        "fs": FS,
        "n_fft": N_FFT,
        "hop_len": HOP_LEN,
        "d": D,
        "c": C,
        "train_seg_samples": TRAIN_SEG_SAMPLES,
        # Physics Parameters
        "freq_cutoff_hz": 500,  # Below this, trust physics less
        "low_freq_weight": 0.1, # Weight for loss < 500Hz
        "high_freq_weight": 1.0 # Weight for loss > 500Hz
    }
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("[WorldBuilding] Saved config.json with Physics Parameters")

# --- 3. Mixing Logic (For Test Set Generation) ---
def calculate_far_field_delays(azimuth_deg, d, c):
    theta_rad = np.deg2rad(azimuth_deg)
    tau_m1 = (d / 2) * np.cos(theta_rad) / c
    tau_m2 = (d / 2) * np.cos(theta_rad - np.pi) / c
    return tau_m1, tau_m2

def apply_frac_delay(y, delay_sec, fs):
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    return np.fft.irfft(y_fft * phase_shift, n=n)

def load_resample(path):
    y, sr = sf.read(path, dtype='float32')
    if len(y.shape) > 1: y = np.mean(y, axis=1)
    if sr != FS: y = librosa.resample(y, orig_sr=sr, target_sr=FS)
    return y

def mix_and_save(files, prefix):
    if len(files) < 3: return
    
    sources = [
        {'f': files[0], 'a': ANGLE_TARGET, 'r': 'Target'},
        {'f': files[1], 'a': ANGLE_INTERFERER_A, 'r': 'Int'},
        {'f': files[2], 'a': ANGLE_INTERFERER_B, 'r': 'Int'}
    ]
    
    raws = [load_resample(s['f']) for s in sources]
    max_l = max(len(r) for r in raws)
    raws = [np.pad(r, (0, max_l - len(r))) for r in raws]

    m1 = np.zeros(max_l)
    m2 = np.zeros(max_l)
    tgt_ref = np.zeros(max_l)
    int_ref = np.zeros(max_l)

    for i, s in enumerate(sources):
        t1, t2 = calculate_far_field_delays(s['a'], D, C)
        s_m1 = apply_frac_delay(raws[i], t1, FS)
        s_m2 = apply_frac_delay(raws[i], t2, FS)
        m1 += s_m1; m2 += s_m2
        if s['r'] == 'Target': tgt_ref += s_m1
        else: int_ref += s_m1

    mix = np.stack([m1, m2], axis=1)
    norm = np.max(np.abs(mix)) + 1e-9
    
    sf.write(f"mixture_{prefix}.wav", mix / norm, FS)
    sf.write(f"target_ref_{prefix}.wav", tgt_ref / norm, FS)
    # Added interference reference saver
    sf.write(f"interf_ref_{prefix}.wav", int_ref / norm, FS)
    
    print(f"[WorldBuilding] Created mixture_{prefix}.wav, target_ref_{prefix}.wav, and interf_ref_{prefix}.wav")

def main():
    save_config()
    try:
        print("[WorldBuilding] Locating dataset...")
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
    except:
        home = os.path.expanduser("~")
        wav_path = os.path.join(home, ".cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset/versions/1/LJSpeech-1.1/wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
        
    if not files: return
    random.shuffle(files)
    mix_and_save(files[:3], "TEST") # Only need Test set here, Training is on-the-fly

if __name__ == "__main__":
    main()