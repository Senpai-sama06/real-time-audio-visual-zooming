import numpy as np
import scipy.signal
import soundfile as sf
import os
import kagglehub
import glob
import random
import librosa

# --- NEW: Import Core Physics and Helpers from rt_av_zoom.core.world ---
# The physics constants (C, FS, D) and core functions 
# (calculate_far_field_delays, apply_frac_delay, load_audio_and_resample)
# are defined in rt_av_zoom.core.world. We import them here to reuse the logic.
from rt_av_zoom.core.world import (
    C, 
    FS, 
    D, 
    calculate_far_field_delays, 
    apply_frac_delay, 
    load_audio_and_resample
)

# --- Constants ---
# D, C, and FS are now imported, so we comment out the local definitions.
# C = 343.0
# FS = 16000
# D = 0.04 # NOTE: The D value is 0.01 in core/world.py now. The imported D will be used!

# --- Physics (Functions commented out as they are now imported) ---
# def calculate_far_field_delays(azimuth_deg, d, c):
#     ...
# def apply_frac_delay(y, delay_sec, fs):
#     ...
# def load_audio_and_resample(file_path, target_fs):
#     ...

def main():
    print("--- World Builder: 10-Source Crowded Room (Using Core Physics) ---")

    # 1. Get Data
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        all_wav_files = glob.glob(os.path.join(wav_path, "*.wav"))
    except:
        print("Dataset error.")
        return

    # Pick 10 distinct files
    if len(all_wav_files) < 10:
        print("Not enough files.")
        return
        
    selected_files = random.sample(all_wav_files, 10)
    
    # 2. Setup Sources
    sources = []
    
    # Source 0 is ALWAYS Target at 90
    sources.append({
        'file': selected_files[0],
        'angle': 90.0,
        'role': 'Target'
    })
    
    # Sources 1-9 are Interferers at random angles
    # We pick angles between 10 and 170 to avoid pure endfire issues
    random_angles = np.linspace(10, 170, 9) 
    
    for i in range(9):
        sources.append({
            'file': selected_files[i+1],
            'angle': random_angles[i],
            'role': f'Interferer_{i+1}'
        })

    # 3. Load & Pad (Uses imported load_audio_and_resample)
    audio_data = []
    max_len = 0
    print("Loading 10 files...")
    for s in sources:
        # FS is imported and used here
        y = load_audio_and_resample(s['file'], FS) 
        audio_data.append(y)
        max_len = max(max_len, len(y))
        
    # Pad
    for i in range(10):
        if len(audio_data[i]) < max_len:
            audio_data[i] = np.pad(audio_data[i], (0, max_len - len(audio_data[i])))

    # 4. Mix
    y_mic1 = np.zeros(max_len, dtype=np.float32)
    y_mic2 = np.zeros(max_len, dtype=np.float32)
    y_ref_target = np.zeros(max_len, dtype=np.float32)
    y_ref_interference = np.zeros(max_len, dtype=np.float32)

    print("Mixing...")
    for i, s in enumerate(sources):
        print(f"  > Placing {s['role']} at {s['angle']:.1f} degrees")
        
        # Uses imported calculate_far_field_delays, D, and C
        tau1, tau2 = calculate_far_field_delays(s['angle'], 0.0, D, C) 
        
        # Uses imported apply_frac_delay and FS
        s_m1 = apply_frac_delay(audio_data[i], tau1, FS) 
        s_m2 = apply_frac_delay(audio_data[i], tau2, FS)
        
        y_mic1 += s_m1
        y_mic2 += s_m2
        
        if s['role'] == 'Target':
            y_ref_target += s_m1
        else:
            y_ref_interference += s_m1 

    # Normalize 
    mixture = np.stack([y_mic1, y_mic2], axis=1)
    mixture /= (np.max(np.abs(mixture)) + 1e-6)
    
    y_ref_target /= (np.max(np.abs(y_ref_target)) + 1e-6)
    y_ref_interference /= (np.max(np.abs(y_ref_interference)) + 1e-6)
    
    # 5. Save (Kept original names for backwards compatibility)
    sf.write("mixture_3_sources.wav", mixture, FS)
    sf.write("target_reference.wav", y_ref_target, FS)
    sf.write("interference_reference.wav", y_ref_interference, FS)
    
    print("\nDone. Files saved.")

if __name__ == "__main__":
    main()