import numpy as np
import soundfile as sf
import os
import kagglehub
import glob
import random
import librosa

# --- NEW: Import Core Physics and Helpers from rt_av_zoom.core.world ---
# We import the constants and functions defined in the core simulation module
from rt_av_zoom.core.world import (
    C, 
    FS, 
    D, 
    calculate_far_field_delays, 
    apply_frac_delay, 
    load_audio_and_resample # Added for consistent audio loading
)

# --- Constants (Now imported) ---
# C = 343.0
# FS = 16000
# D = 0.04 # This value is now imported from core.world

# --- Physics (Functions commented out as they are now imported) ---
# def calculate_far_field_delays(azimuth_deg, d, c):
#     ...
# def apply_frac_delay(y, delay_sec, fs):
#     ...
# def load_audio_and_resample(file_path, target_fs):
#     ...

def main():
    print("--- WORLD CALIBRATION: SINGLE SOURCE @ 90 (Using Core Physics) ---")
    
    # Get one file
    path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
    wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
    file_path = glob.glob(os.path.join(wav_path, "*.wav"))[0]
    
    # Load (Using imported load_audio_and_resample and FS)
    y = load_audio_and_resample(file_path, FS)
    y = y[:FS*3] # 3 seconds
    
    # --- PHYSICS: 1 Source at 90 Degrees ---
    # At 90 degrees, the delays should be EXACTLY ZERO.
    # This means Mic 1 and Mic 2 should be IDENTICAL.
    # Uses imported calculate_far_field_delays, D, and C
    tau_m1, tau_m2 = calculate_far_field_delays(90.0, 0.0, D, C) # Added 0.0 elevation for consistency
    
    print(f"Delays for 90 deg: M1={tau_m1*1e6:.2f}us, M2={tau_m2*1e6:.2f}us")
    
    # Uses imported apply_frac_delay and FS
    y_m1 = apply_frac_delay(y, tau_m1, FS)
    y_m2 = apply_frac_delay(y, tau_m2, FS)
    
    mixture = np.stack([y_m1, y_m2], axis=1)
    sf.write("mixture_3_sources.wav", mixture, FS) # Overwrite with clean test
    print("Saved 'mixture_3_sources.wav' (Actually just 1 source at 90)")

if __name__ == "__main__":
    main()