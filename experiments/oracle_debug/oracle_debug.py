import numpy as np
import scipy.signal
import soundfile as sf
import os
import matplotlib.pyplot as plt
import matplotlib as mlt
# mlt.use("TkAgg") # Removed line if you are running in a notebook/non-GUI environment

# --- Import Core Constants and Helpers ---
# We reuse the constants and steering vector function from the core MVDR module
from rt_av_zoom.core.masked_mvdr import (
    get_steering_vector, 
    D, 
    C, 
    N_MICS, 
    FS, 
    N_FFT, 
    N_HOP
)

# --- Constants ---
# ANGLE_TARGET is still needed locally as it's a specific experiment parameter
ANGLE_TARGET = 90.0
SIGMA = 1 # Specific to this debug/oracle run

def main():
    print("--- ORACLE TEST: Can the code theoretically work? ---")
    
    # 1. Load The "Answer Key" Files
    if not os.path.exists("target_reference.wav") or not os.path.exists("interference_reference.wav"):
        print("Error: Reference files missing. Run world.py first.")
        return

    y_mix, _ = sf.read("mixture_3_sources.wav", dtype='float32') # [Samples, 2]
    y_mix = y_mix.T 
    
    s_tgt_ref, _ = sf.read("target_reference.wav", dtype='float32')
    s_int_ref, _ = sf.read("interference_reference.wav", dtype='float32')
    
    # 2. Compute STFTs (Using imported N_FFT and N_HOP)
    f, t, Y_mix = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_tgt = scipy.signal.stft(s_tgt_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_int = scipy.signal.stft(s_int_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP)

    print("Oracle Mask created directly from ground truth files.")

    # 3. Construct the ORACLE MASK (The Cheat)
    mag_tgt = np.abs(S_tgt)
    mag_int = np.abs(S_int)
    
    # Ideal Binary Mask (IBM)
    mask_noise = np.where(mag_int > mag_tgt, 1.0, 0.0)

    # 4. Run the Pipeline (Exact same logic as before)
    n_channels, n_freqs, n_frames = Y_mix.shape
    R_noise = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    
    print("Computing Covariance...")
    for f_idx in range(n_freqs):
        m_f = mask_noise[f_idx, :] 
        Y_f = Y_mix[:, f_idx, :] 
        Y_weighted = Y_f * np.sqrt(m_f) 
        R_noise[f_idx] = (Y_weighted @ Y_weighted.conj().T) / (np.sum(m_f) + 1e-6)

    print("Running MVDR...")
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    for f_idx in range(n_freqs):
        if f[f_idx] < 100: continue 
        R = R_noise[f_idx] + SIGMA * np.eye(N_MICS) # Use imported N_MICS
        
        # Use imported get_steering_vector and constants
        d = get_steering_vector(ANGLE_TARGET, f[f_idx], D, C) 
        
        try:
            w = np.linalg.solve(R, d)
            w /= (d.conj().T @ w + 1e-10)
        except:
            w = np.array([[1], [0]])
        S_mvdr[f_idx, :] = w.conj().T @ Y_mix[:, f_idx, :]

    print("Applying Aggressive Post-Filter...")
    # INVERT Mask: 1 = Target, 0 = Noise
    mask_target = 1.0 - mask_noise 
    
    gain_filter = mask_target 
    # plt.imshow(np.abs(S_final)) # Plotting code removed for non-GUI execution stability
    # plt.show()
    
    S_final = S_mvdr * gain_filter
    
    # 5. Reconstruction
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    s_out /= np.max(np.abs(s_out))
    
    # sf.write("output_oracle.wav", s_out, FS)
    # print("Saved 'output_oracle.wav'.")

if __name__ == "__main__":
    main()