import numpy as np
import scipy.signal
import soundfile as sf
import os
import matplotlib.pyplot as plt
import matplotlib as mlt
mlt.use("TkAgg")


# --- Constants ---
FS = 16000
N_MICS = 2
N_FFT = 512
N_HOP = 256
D = 0.04   # Make sure this matches your world.py!
C = 343.0
ANGLE_TARGET = 90.0
SIGMA = 1

def get_steering_vector(angle_deg, f, d, c):
    theta_rad = np.deg2rad(angle_deg)
    phi_rad = 0.0 
    tau_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi) / c
    omega = 2 * np.pi * f
    d_vec = np.array([[np.exp(-1j * omega * tau_m1)], [np.exp(-1j * omega * tau_m2)]], dtype=complex)
    return d_vec

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
    
    # 2. Compute STFTs
    f, t, Y_mix = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_tgt = scipy.signal.stft(s_tgt_ref, fs=FS, nperseg=N_FFT, noverlap=N_FFT//2) # Ensure overlap matches (N_HOP vs N_FFT//2)
    _, _, S_int = scipy.signal.stft(s_int_ref, fs=FS, nperseg=N_FFT, noverlap=N_FFT//2)
    
    # **Sanity Check on STFT shapes**
    # Often scipy defaults hop size differently if not specified.
    # We explicitly used N_HOP=256 above.
    _, _, S_tgt = scipy.signal.stft(s_tgt_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_int = scipy.signal.stft(s_int_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP)

    # 3. Construct the ORACLE MASK (The Cheat)
    # Mask = 1 if Interference > Target (i.e., this bin is Noise)
    mag_tgt = np.abs(S_tgt)
    mag_int = np.abs(S_int)
    
    # Ideal Binary Mask (IBM)
    mask_noise = np.where(mag_int > mag_tgt, 1.0, 0.0)
    
    print("Oracle Mask created directly from ground truth files.")

    # 4. Run the Pipeline (Exact same logic as before)
    n_channels, n_freqs, n_frames = Y_mix.shape
    R_noise = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    
    print("Computing Covariance...")
    for f_idx in range(n_freqs):
        m_f = mask_noise[f_idx, :] 
        Y_f = Y_mix[:, f_idx, :] 
        Y_weighted = Y_f * np.sqrt(m_f) 
        # Add tiny epsilon to normalization to avoid divide by zero
        R_noise[f_idx] = (Y_weighted @ Y_weighted.conj().T) / (np.sum(m_f) + 1e-6)

    print("Running MVDR...")
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    for f_idx in range(n_freqs):
        if f[f_idx] < 100: continue 
        R = R_noise[f_idx] + SIGMA * np.eye(n_channels)
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
    
    # --- REMOVE THE SAFETY FLOOR ---
    # Old Code: gain_filter = np.maximum(mask_target, 0.1)
    # New Code: Allow it to go to zero!
    gain_filter = mask_target 
    import matplotlib.pyplot as plt
    # import numpy as np
    S_final = S_mvdr * gain_filter
    plt.imshow(np.abs(S_final))
    plt.show()
    # 5. Reconstruction
    # _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    # s_out /= np.max(np.abs(s_out))
    
    # sf.write("output_oracle.wav", s_out, FS)
    # print("Saved 'output_oracle.wav'.")
    # print("Run validate.py on this file. If SIR < 15dB, the beamformer math is broken.")

if __name__ == "__main__":
    main()