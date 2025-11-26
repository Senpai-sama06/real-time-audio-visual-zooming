import numpy as np
import scipy.signal
import soundfile as sf
import os
import kagglehub
import glob
import random
import librosa
from scipy.signal import stft, istft

# --- 1. Constants ---
C = 343.0           # Speed of sound (m/s)
FS = 16000          # Target sample rate (Hz)
D = 0.04            # Mic spacing (m)
ANGLE_TARGET = 90.0
ANGLE_INTERFERER_A = 40.0
ANGLE_INTERFERER_B = 130.0

# --- 2. Core Physics Functions (Reusable) ---
def calculate_far_field_delays(azimuth_deg, d, c):
    theta_rad = np.deg2rad(azimuth_deg)
    # Mic 1 (Right) and Mic 2 (Left) relative to origin
    tau_m1 = (d / 2) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(theta_rad - np.pi) / c
    return tau_m1, tau_m2

def apply_frac_delay(y, delay_sec, fs):
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    y_delayed = np.fft.irfft(y_fft * phase_shift, n=n)
    return y_delayed

def load_audio_and_resample(file_path, target_fs):
    y, orig_fs = sf.read(file_path, dtype='float32')
    if len(y.shape) > 1: y = np.mean(y, axis=1)
    if orig_fs != target_fs: y = librosa.resample(y, orig_sr=orig_fs, target_sr=target_fs)
    return y

# --- 3. Main Script ---
def main():
    print("--- 1. World Builder: Generating 3-Source Test Set ---")

    # --- Step 1: Get Data ---
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        all_wav_files = glob.glob(os.path.join(wav_path, "*.wav"))
        selected_files = random.sample(all_wav_files, 3)
    except Exception as e:
        print(f"Download failed: {e}. Please ensure Kaggle API is configured.")
        return
    
    # --- Step 2: Define the "Impossible" Setup ---
    sources_setup = [
        {'file': selected_files[0], 'azimuth_deg': ANGLE_TARGET, 'role': 'Target'},
        {'file': selected_files[1], 'azimuth_deg': ANGLE_INTERFERER_A, 'role': 'InterfererA'},
        {'file': selected_files[2], 'azimuth_deg': ANGLE_INTERFERER_B, 'role': 'InterfererB'}
    ]
    
    # --- Step 3: Load, Pad, and Mix ---
    audio_sources = {}
    max_len = 0
    for source in sources_setup:
        audio = load_audio_and_resample(source['file'], FS)
        audio_sources[source['file']] = audio
        max_len = max(max_len, len(audio))
            
    for k, v in audio_sources.items():
        if len(v) < max_len: audio_sources[k] = np.pad(v, (0, max_len - len(v)))

    y_mic1 = np.zeros(max_len, dtype=np.float32)
    y_mic2 = np.zeros(max_len, dtype=np.float32)
    y_target_ref = np.zeros(max_len, dtype=np.float32)
    y_interferer_ref = np.zeros(max_len, dtype=np.float32)

    print(f"Mixing {max_len/FS:.2f} seconds of audio...")
    for source in sources_setup:
        s_audio = audio_sources[source['file']]
        tau_m1, tau_m2 = calculate_far_field_delays(source['azimuth_deg'], D, C)
        
        s_delayed_m1 = apply_frac_delay(s_audio, tau_m1, FS)
        
        y_mic1 += s_delayed_m1
        y_mic2 += apply_frac_delay(s_audio, tau_m2, FS) # Mic 2 signal
        
        if source['role'] == 'Target':
            y_target_ref += s_delayed_m1
        else:
            y_interferer_ref += s_delayed_m1

    # Normalize and Save
    mixture = np.stack([y_mic1, y_mic2], axis=1)
    mixture /= np.max(np.abs(mixture))
    y_target_ref /= (np.max(np.abs(y_target_ref)) + 1e-6)
    y_interferer_ref /= (np.max(np.abs(y_interferer_ref)) + 1e-6)
    
    sf.write("mixture_3_sources.wav", mixture, FS)
    sf.write("target_reference.wav", y_target_ref, FS)
    sf.write("interference_reference.wav", y_interferer_ref, FS)
    
    print(f"\nSaved 3 files. Ready for training and validation.")

if __name__ == "__main__":
    main()