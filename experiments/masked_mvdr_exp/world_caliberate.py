import numpy as np
import soundfile as sf
import os
import kagglehub
import glob
import random
import librosa

# --- Constants ---
C = 343.0
FS = 16000
D = 0.08

def calculate_far_field_delays(azimuth_deg, d, c):
    theta_rad = np.deg2rad(azimuth_deg)
    # Simple Linear Array Geometry
    # Mic 1 at +d/2, Mic 2 at -d/2
    # If source is at 90 (Broadside), cos(90) = 0 -> Delays are 0.
    # If source is at 0 (Endfire), Mic 1 is closer (-delay), Mic 2 is farther (+delay).
    
    # Delay relative to center:
    # tau = - (x_mic * cos(theta)) / c
    
    tau_m1 = - ( (d/2) * np.cos(theta_rad) ) / c
    tau_m2 = - ( (-d/2) * np.cos(theta_rad) ) / c
    
    return tau_m1, tau_m2

def apply_frac_delay(y, delay_sec, fs):
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    y_delayed = np.fft.irfft(y_fft * phase_shift, n=n)
    return y_delayed

def main():
    print("--- WORLD CALIBRATION: SINGLE SOURCE @ 90 ---")
    
    # Get one file
    path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
    wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
    file_path = glob.glob(os.path.join(wav_path, "*.wav"))[0]
    
    # Load
    y, _ = sf.read(file_path, dtype='float32')
    if len(y.shape) > 1: y = np.mean(y, axis=1)
    y = librosa.resample(y, orig_sr=22050, target_sr=FS)
    y = y[:FS*3] # 3 seconds
    
    # --- PHYSICS: 1 Source at 90 Degrees ---
    # At 90 degrees, the delays should be EXACTLY ZERO.
    # This means Mic 1 and Mic 2 should be IDENTICAL.
    tau_m1, tau_m2 = calculate_far_field_delays(90.0, D, C)
    
    print(f"Delays for 90 deg: M1={tau_m1*1e6:.2f}us, M2={tau_m2*1e6:.2f}us")
    
    y_m1 = apply_frac_delay(y, tau_m1, FS)
    y_m2 = apply_frac_delay(y, tau_m2, FS)
    
    mixture = np.stack([y_m1, y_m2], axis=1)
    sf.write("mixture_3_sources.wav", mixture, FS) # Overwrite with clean test
    print("Saved 'mixture_3_sources.wav' (Actually just 1 source at 90)")

if __name__ == "__main__":
    main()