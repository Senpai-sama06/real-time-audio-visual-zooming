import numpy as np
import soundfile as sf
import os
import kagglehub
import glob
import random
import librosa
import argparse

# --- Constants ---
C = 343.0
FS = 16000
D = 0.08

def calculate_far_field_delays(azimuth_deg, d, c):
    theta_rad = np.deg2rad(azimuth_deg)
    tau_m1 = (d / 2) * np.cos(0) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(0) * np.cos(theta_rad - np.pi) / c
    return tau_m1, tau_m2

def apply_frac_delay(y, delay_sec, fs):
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    return np.fft.irfft(y_fft * phase_shift, n=n)

def load_audio_and_resample(file_path, target_fs):
    y, orig_fs = sf.read(file_path, dtype='float32')
    if len(y.shape) > 1: y = np.mean(y, axis=1)
    if orig_fs != target_fs: y = librosa.resample(y, orig_sr=orig_fs, target_sr=target_fs)
    return y

def main():
    # --- ARGUMENT PARSING (Crucial for Benchmark) ---
    parser = argparse.ArgumentParser(description="World Builder: 10-Source Stress Test")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--index", type=str, required=True, help="Run Index (e.g. 00, 01)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Get Data
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        all_wav_files = glob.glob(os.path.join(wav_path, "*.wav"))
    except:
        print("Dataset error.")
        return

    if len(all_wav_files) < 10:
        print("Not enough files.")
        return
        
    selected_files = random.sample(all_wav_files, 10)
    
    # 2. Setup Sources
    sources = []
    # Source 0 is ALWAYS Target at 90
    sources.append({'file': selected_files[0], 'angle': 90.0, 'role': 'Target'})
    
    # Sources 1-9 are Interferers
    random_angles = np.linspace(10, 170, 9) 
    for i in range(9):
        sources.append({'file': selected_files[i+1], 'angle': random_angles[i], 'role': f'Interferer_{i+1}'})

    # 3. Load & Pad
    audio_data = []
    max_len = 0
    for s in sources:
        y = load_audio_and_resample(s['file'], FS)
        audio_data.append(y)
        max_len = max(max_len, len(y))
        
    for i in range(10):
        if len(audio_data[i]) < max_len:
            audio_data[i] = np.pad(audio_data[i], (0, max_len - len(audio_data[i])))

    # 4. Mix
    y_mic1 = np.zeros(max_len, dtype=np.float32)
    y_mic2 = np.zeros(max_len, dtype=np.float32)
    y_tgt_ref = np.zeros(max_len, dtype=np.float32)
    y_int_ref = np.zeros(max_len, dtype=np.float32)

    for i, s in enumerate(sources):
        tau1, tau2 = calculate_far_field_delays(s['angle'], D, C)
        s_m1 = apply_frac_delay(audio_data[i], tau1, FS)
        s_m2 = apply_frac_delay(audio_data[i], tau2, FS)
        
        y_mic1 += s_m1
        y_mic2 += s_m2
        
        if s['role'] == 'Target':
            y_tgt_ref += s_m1
        else:
            y_int_ref += s_m1 

    # Normalize
    mixture = np.stack([y_mic1, y_mic2], axis=1)
    scale = max(np.max(np.abs(mixture)), 1e-9)
    mixture /= scale
    y_tgt_ref /= scale
    y_int_ref /= scale
    
    # 5. Save with Standardized Naming for Benchmark
    # Note: args.index is passed as a string (e.g., "00")
    f_mix = os.path.join(args.output_dir, f"{args.index}_mixture.wav")
    f_tgt = os.path.join(args.output_dir, f"{args.index}_target_reference.wav")
    f_int = os.path.join(args.output_dir, f"{args.index}_interference_reference.wav")

    sf.write(f_mix, mixture, FS)
    sf.write(f_tgt, y_tgt_ref, FS)
    sf.write(f_int, y_int_ref, FS)

if __name__ == "__main__":
    main()