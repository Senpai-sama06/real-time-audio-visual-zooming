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
FS = 16000          # Our *target* sample rate (Hz)
D = 0.04            # Mic spacing (m)

# --- 2. Core Physics Functions (Unchanged) ---

def calculate_far_field_delays(azimuth_deg, elevation_deg, d, c):
    """Calculates the TDoA (in seconds) for our 3D far-field model."""
    theta_rad = np.deg2rad(azimuth_deg)
    phi_rad = np.deg2rad(elevation_deg)
    
    path_diff_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0)
    path_diff_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi)

    tau_m1 = path_diff_m1 / c
    tau_m2 = path_diff_m2 / c
    
    return tau_m1, tau_m2

def apply_frac_delay(y, delay_sec, fs):
    """Applies a fractional delay to a signal 'y' using the FFT-shift method."""
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    y_delayed_fft = y_fft * phase_shift
    y_delayed = np.fft.irfft(y_delayed_fft, n=n)
    return y_delayed

# --- 3. Helper for Loading Audio (Unchanged) ---

def load_audio_and_resample(file_path, target_fs):
    """
    Loads an audio file, converts to mono, and resamples
    to the target_fs (16kHz).
    """
    y, orig_fs = sf.read(file_path, dtype='float32')
    
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    if orig_fs != target_fs:
        y = librosa.resample(y, orig_sr=orig_fs, target_sr=target_fs)
        
    return y

# --- 4. Main Script (MODIFIED AS REQUESTED) ---

def main():
    print("--- 1. World Builder (2-Source Test) ---")

    # --- Step 1: Download the dataset ---
    print("Downloading Kaggle dataset (if needed)...")
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return
        
    # --- Step 2: Find all the .wav files ---
    wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
    all_wav_files = glob.glob(os.path.join(wav_path, "*.wav"))
    
    if len(all_wav_files) < 2:
        print("Error: Not enough .wav files found in dataset.")
        return
        
    # --- Step 3: Randomly pick 2 files ---
    print("Randomly selecting 2 source files...")
    selected_files = random.sample(all_wav_files, 2)
    
    # --- YOUR NEW 2-SOURCE EXPERIMENTAL SETUP ---
    sources_setup = [
        {
            'file': selected_files[0],
            'azimuth_deg': 90.0,    # Broadside (Target)
            'elevation_deg': 0.0,
            'role': 'Target'
        },
        {
            'file': selected_files[1],
            'azimuth_deg':80.0,    # Far-field (Interferer)
            'elevation_deg': 0.0,
            'role': 'Interferer'
        }
    ]
    
    # --- Load, Resample, and Pad Audio ---
    audio_sources = {}
    max_len = 0
    
    print(f"Loading and resampling audio to {FS}Hz...")
    for source in sources_setup:
        file_name = os.path.basename(source['file'])
        print(f"  > Loading {file_name} as {source['role']}")
        
        audio = load_audio_and_resample(source['file'], target_fs=FS) 
        audio_sources[source['file']] = audio
        if len(audio) > max_len:
            max_len = len(audio)
            
    # Pad all sources to the same max_len
    for file_name in audio_sources:
        y = audio_sources[file_name]
        if len(y) < max_len:
            audio_sources[file_name] = np.pad(y, (0, max_len - len(y)))

    print(f"All sources loaded and padded to {max_len} samples.")

    # --- Initialize output channels ---
    y_mic1 = np.zeros(max_len, dtype=np.float32)
    y_mic2 = np.zeros(max_len, dtype=np.float32)

    # --- Build the Mixture ---
    for source in sources_setup:
        print(f"  Adding {source['role']} at (Azi={source['azimuth_deg']}, Ele={source['elevation_deg']})")
        
        s_audio = audio_sources[source['file']]
        
        tau_m1, tau_m2 = calculate_far_field_delays(
            source['azimuth_deg'], source['elevation_deg'], D, C
        )
        
        s_delayed_m1 = apply_frac_delay(s_audio, tau_m1, FS)
        s_delayed_m2 = apply_frac_delay(s_audio, tau_m2, FS)
        
        y_mic1 += s_delayed_m1
        y_mic2 += s_delayed_m2

    # --- 5. Finalize and Save ---
    mixture = np.stack([y_mic1, y_mic2], axis=1)
    
    max_val = np.max(np.abs(mixture))
    if max_val > 0:
        mixture /= max_val
    
    # Save to a new file to not break your old experiment
    output_filename = "mixture_2_sources.wav"
    sf.write(output_filename, mixture, FS)
    
    print(f"\nDone. '{output_filename}' created.")
    print(f"This file is 16kHz and ready for your MVDR script.")

if __name__ == "__main__":
    main()