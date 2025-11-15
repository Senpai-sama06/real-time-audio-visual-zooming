import numpy as np
import scipy.signal
import scipy.linalg  # We need the GEV solver
import soundfile as sf
import os
import matplotlib.pyplot as plt

# --- 1. Constants ---
FS = 16000
N_MICS = 2
N_FFT = 512
N_HOP = 256

# --- 2. Main Script ---
def main():
    print("--- ORACLE TEST (GEV / Max-SNR Engine) ---")
    
    # 1. Load All Files
    try:
        y_mix, _ = sf.read("test_mixture.wav", dtype='float32')
        y_mix = y_mix.T 
        s_tgt_ref, _ = sf.read("test_target_ref.wav", dtype='float32')
        s_int_ref, _ = sf.read("test_interferer_ref.wav", dtype='float32')
    except FileNotFoundError as e:
        print(f"Error: {e}. Run world.py first.")
        return

    # 2. Compute all STFTs
    f, t, Y_mix = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_tgt = scipy.signal.stft(s_tgt_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_int = scipy.signal.stft(s_int_ref, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    n_channels, n_freqs, n_frames = Y_mix.shape

    # 3. Construct the ORACLE MASKS
    print("Building Oracle Masks...")
    mag_tgt = np.abs(S_tgt)
    mag_int = np.abs(S_int)
    
    # We create two masks now
    mask_target = np.where(mag_tgt > mag_int, 1.0, 0.0)
    mask_noise = 1.0 - mask_target
    
    # 4. Calculate *Two* Oracle Covariance Matrices
    print("Computing Oracle R_speech and R_noise...")
    R_speech = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    R_noise = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    
    # Tiny value for stability
    stability_eps = 1e-6 * np.eye(n_channels)
    
    for f_idx in range(n_freqs):
        # Data for this frequency bin
        Y_f = Y_mix[:, f_idx, :]  # [2, Time]
        
        # --- Build R_speech ---
        m_s = mask_target[f_idx, :]
        Y_s_weighted = Y_f * np.sqrt(m_s)
        R_s = (Y_s_weighted @ Y_s_weighted.conj().T) / (np.sum(m_s) + 1e-6)
        
        # --- Build R_noise ---
        m_n = mask_noise[f_idx, :]
        Y_n_weighted = Y_f * np.sqrt(m_n)
        R_n = (Y_n_weighted @ Y_n_weighted.conj().T) / (np.sum(m_n) + 1e-6)
        
        R_speech[f_idx] = R_s + stability_eps
        R_noise[f_idx] = R_n + stability_eps

    # 5. Beamforming Loop (Solving GEV)
    print("Solving GEV for each frequency...")
    S_gev_stft = np.zeros((n_freqs, n_frames), dtype=complex)
    
    for f_idx in range(n_freqs):
        if f[f_idx] < 100: continue
            
        R_s = R_speech[f_idx]
        R_n = R_noise[f_idx]
        
        try:
            # --- This is the GEV "Engine" ---
            # Solve R_s * w = lambda * R_n * w
            eigvals, eigvecs = scipy.linalg.eig(R_s, R_n)
            
            # The filter w is the eigenvector with the largest eigenvalue
            w = eigvecs[:, np.argmax(np.real(eigvals))]
            w = w.reshape(n_channels, 1)
            
        except scipy.linalg.LinAlgError:
            w = np.array([[1], [0]]) # Fallback
            
        # Apply the GEV filter
        S_gev_stft[f_idx, :] = w.conj().T @ Y_mix[:, f_idx, :]

    # 6. Apply Spectral Post-Filter (Just like the MVDR Oracle)
    # We multiply the spatially filtered signal by the target mask
    print("Applying Oracle Post-Filter...")
    S_final_stft = S_gev_stft * mask_target
        
    # 7. Reconstruction
    _, s_out = scipy.signal.istft(S_final_stft, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    s_out /= (np.max(np.abs(s_out)) + 1e-6)
    
    sf.write("output_oracle_gev.wav", s_out, FS)
    print("\nDone. Saved 'output_oracle_gev.wav'.")
    print("Now run validate.py on this file and compare to the 36dB MVDR result.")

if __name__ == "__main__":
    main()