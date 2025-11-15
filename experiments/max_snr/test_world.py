import numpy as np
import scipy.signal
import soundfile as sf
import os
import kagglehub
import glob
import random
import librosa

# --- 1. Constants ---
C = 343.0           # Speed of sound (m/s)
FS = 16000          # Target sample rate (Hz)
D = 0.04            # Mic spacing (m)

# --- 2. Core Physics Functions ---

def calculate_far_field_delays(azimuth_deg, d, c):
    """Calculates TDoA for our 2-mic array (mics on x-axis)."""
    theta_rad = np.deg2rad(azimuth_deg)
    phi_rad = 0.0 # Elevation
    
    # Mic 1 (r=d/2, theta=0)
    tau_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0) / c
    # Mic 2 (r=d/2, theta=pi)
    tau_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi) / c
    
    return tau_m1, tau_m2

def apply_frac_delay(y, delay_sec, fs):
    """Applies a fractional delay using FFT phase shift."""
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    y_delayed = np.fft.irfft(y_fft * phase_shift, n=n)
    return y_delayed

def load_audio_and_resample(file_path, target_fs):
    """Loads, forces mono, and resamples."""
    y, orig_fs = sf.read(file_path, dtype='float32')
    if len(y.shape) > 1: y = np.mean(y, axis=1)
    if orig_fs != target_fs: y = librosa.resample(y, orig_sr=orig_fs, target_sr=target_fs)
    return y

# --- 4. Main Script ---

def main():
    print("--- [Test Arch] World Builder: 3-Source ---")
    
    # --- 1. Get Data ---
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        all_wav_files = glob.glob(os.path.join(wav_path, "*.wav"))
        selected_files = random.sample(all_wav_files, 3)
    except Exception as e:
        print(f"Dataset error: {e}")
        return
    
    # --- 2. Define Sources ---
    sources_setup = [
        {'file': selected_files[0], 'azimuth_deg': 90.0, 'role': 'Target'},
        {'file': selected_files[1], 'azimuth_deg': 80.0, 'role': 'InterfererA'}, # Near target
        {'file': selected_files[2], 'azimuth_deg': 40.0, 'role': 'InterfererB'}  # Far target
    ]
    
    # --- 3. Load & Process ---
    audio_sources = {}
    max_len = 0
    for source in sources_setup:
        audio = load_audio_and_resample(source['file'], FS)
        audio_sources[source['file']] = audio
        max_len = max(max_len, len(audio))
            
    for k, v in audio_sources.items():
        if len(v) < max_len:
            audio_sources[k] = np.pad(v, (0, max_len - len(v)))

    # --- 4. Build Mixture & References ---
    y_mic1 = np.zeros(max_len, dtype=np.float32)
    y_mic2 = np.zeros(max_len, dtype=np.float32)
    y_target_ref = np.zeros(max_len, dtype=np.float32)
    y_interferer_ref = np.zeros(max_len, dtype=np.float32)

    print("Mixing sources...")
    for source in sources_setup:
        s_audio = audio_sources[source['file']]
        tau_m1, tau_m2 = calculate_far_field_delays(source['azimuth_deg'], D, C)
        
        s_m1 = apply_frac_delay(s_audio, tau_m1, FS)
        s_m2 = apply_frac_delay(s_audio, tau_m2, FS)
        
        y_mic1 += s_m1
        y_mic2 += s_m2
        
        if source['role'] == 'Target':
            y_target_ref += s_m1
        else:
            y_interferer_ref += s_m1

    # Normalize
    mixture = np.stack([y_mic1, y_mic2], axis=1)
    mixture /= (np.max(np.abs(mixture)) + 1e-6)
    y_target_ref /= (np.max(np.abs(y_target_ref)) + 1e-6)
    y_interferer_ref /= (np.max(np.abs(y_interferer_ref)) + 1e-6)
    
    # --- 5. Save to NEW Files ---
    sf.write("test_mixture.wav", mixture, FS)
    sf.write("test_target_ref.wav", y_target_ref, FS)
    sf.write("test_interferer_ref.wav", y_interferer_ref, FS)
    
    print(f"\nDone. Test files created ('test_...wav')")

if __name__ == "__main__":
    main()