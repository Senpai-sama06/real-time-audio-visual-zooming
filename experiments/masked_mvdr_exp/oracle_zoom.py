import numpy as np
import scipy.signal
import soundfile as sf
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FS = 16000
N_MICS = 2
N_FFT = 1024
N_HOP = 512
D = 0.08   # Make sure this matches your world.py!
C = 343.0
ANGLE_TARGET = 90.0
SIGMA = 1.0 # High sigma for stability, mask does the heavy lifting
SAVE_DIR = "/home/cse-sdpl/paarth/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/samples"

# --- TUNABLE KNOB ---
# Small value (0.2) = Narrow Beam (Zoom In)
# Large value (1.5) = Wide Beam (Zoom Out)
# USER_ZOOM_KNOB = 0.1
USER_ZOOM_DEG = 90.0   # half-width of angular window in degrees


# --- HELPERS ---

def get_steering_vector(angle_deg, f, d, c):
    theta_rad = np.deg2rad(angle_deg)
    phi_rad = 0.0 
    tau_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi) / c
    omega = 2 * np.pi * f
    d_vec = np.array([[np.exp(-1j * omega * tau_m1)], [np.exp(-1j * omega * tau_m2)]], dtype=complex)
    return d_vec

# Replace compute_geometric_mask in oracle_zoom.py with this function

def compute_geometric_mask(Y_stft, center_angle_deg, zoom_deg, fs, d, c, gamma_min=0.01):
    """
    Frequency-dependent geometric mask. 
    - Y_stft: (n_mics, n_freqs, n_frames)
    - center_angle_deg: steering centre (e.g. 90)
    - zoom_deg: half-width of angular window in degrees (e.g. 30)
    - fs: sample rate
    - d, c: mic spacing and speed of sound
    """
    Y1 = Y_stft[0]   # (n_freqs, n_frames)
    Y2 = Y_stft[1]

    n_freqs = Y1.shape[0]
    # frequencies corresponding to STFT bins: compute same vector as scipy.signal.stft returns
    # You already have f returned from stft in main(); pass it in OR compute approximate freqs:
    freqs = np.fft.rfftfreq(1024, 1.0/fs)[:n_freqs]  # if you used N_FFT=1024

    # expected phase difference for center angle:
    theta0 = np.deg2rad(center_angle_deg)
    # For symmetric two-mic linear array with mic positions +-d/2, time diff tau(theta) = (d/c) * cos(theta)
    tau0 = (d / 2.0) * np.cos(theta0) / c
    # For angle offset +/- zoom_deg we compute min/max phase:
    theta_lo = np.deg2rad(center_angle_deg - zoom_deg)
    theta_hi = np.deg2rad(center_angle_deg + zoom_deg)
    tau_lo = (d / 2.0) * np.cos(theta_lo) / c
    tau_hi = (d / 2.0) * np.cos(theta_hi) / c

    # Precompute expected phase and allowed deviation per frequency
    phi0 = 2.0 * np.pi * freqs * tau0            # expected phase diff at freq
    phi_lo = 2.0 * np.pi * freqs * tau_lo
    phi_hi = 2.0 * np.pi * freqs * tau_hi

    # allowed deviation (take max absolute difference from center)
    eps_f = np.maximum(np.abs(phi_lo - phi0), np.abs(phi_hi - phi0))  # shape (n_freqs,)

    # Observed phase difference
    delta_phi = np.angle(Y1 * np.conj(Y2))  # shape (n_freqs, n_frames)

    # We want the wrapped angular difference between observed and expected
    # broadcast phi0 to (n_freqs, n_frames)
    phi0_mat = phi0[:, None]
    def angle_diff(a, b):
        diff = a - b
        # wrap to [-pi, pi]
        return (diff + np.pi) % (2*np.pi) - np.pi

    dev = np.abs(angle_diff(delta_phi, phi0_mat))  # shape (n_freqs, n_frames)

    # Build mask where observed dev is <= eps_f
    eps_mat = eps_f[:, None] + 1e-12
    M_angle = (dev <= eps_mat).astype(float)

    # Power / voice activity: use combined mic power
    power = (np.abs(Y1)**2 + np.abs(Y2)**2) / 2.0
    power_norm = power / (np.max(power) + 1e-12)
    M_power = (power_norm >= gamma_min).astype(float)

    M_geo = M_angle * M_power
    return M_geo


# --- MAIN ---

def main():
    print(f"--- ORACLE ZOOM TEST (Width={USER_ZOOM_DEG}) ---")
    
    # 1. Load Files
    if not os.path.exists(f"{SAVE_DIR}/target_reference.wav"):
        print("Error: Reference files missing."); return  

    y_mix, _ = sf.read(f"{SAVE_DIR}/mixture_3_sources.wav", dtype='float32')
    y_mix = y_mix.T 
    s_tgt, _ = sf.read(f"{SAVE_DIR}/target_reference.wav", dtype='float32')
    s_int, _ = sf.read(f"{SAVE_DIR}/interference_reference.wav", dtype='float32')
    
    # 2. Compute STFTs
    f, t, Y_mix = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_tgt = scipy.signal.stft(s_tgt, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_int = scipy.signal.stft(s_int, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    n_channels, n_freqs, n_frames = Y_mix.shape

    # 3. Step A: The "Perfect AI" (Oracle Mask)
    mag_tgt = np.abs(S_tgt)
    mag_int = np.abs(S_int)
    ratio = mag_tgt / (mag_int + 1e-9)
    M_oracle = (ratio >= 1.2).astype(float)  # require 20% higher energy from target

    
    # 4. Step B: The "Zoom Knob" (Geometric Mask)
    M_geo = compute_geometric_mask(Y_mix, ANGLE_TARGET, zoom_deg=USER_ZOOM_DEG, fs=FS, d=D, c=C, gamma_min=0.01)

    
    # 5. Step C: Combine (Refine)
    # Logic: Target must be "Speech" (Oracle) AND "In Window" (Geometric)
    M_target_final = M_oracle * M_geo
    
    # Noise is everything else
    M_noise = 1.0 - M_target_final

    # 6. MVDR Beamforming
    print("Computing Covariance & Beamforming...")
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    
    for f_idx in range(n_freqs):
        if f_idx < 5: continue 
        
        # Calculate Noise Covariance
        m_f = M_noise[f_idx, :] 
        Y_f = Y_mix[:, f_idx, :] 
        Y_weighted = Y_f * np.sqrt(m_f) 
        R_n = (Y_weighted @ Y_weighted.conj().T) / (np.sum(m_f) + 1e-6)
        
        # Diagonal Loading
        R_n += SIGMA * np.eye(n_channels)
        
        # Weights
        d = get_steering_vector(ANGLE_TARGET, f[f_idx], D, C)
        try:
            w = np.linalg.solve(R_n, d)
            w /= (d.conj().T @ w + 1e-10)
        except:
            w = np.array([[1], [0]])
            
        S_mvdr[f_idx, :] = w.conj().T @ Y_mix[:, f_idx, :]

    # 7. Post-Filter
    print("Applying Post-Filter...")
    gain = np.maximum(M_target_final, 0.05)
    # gain = 1
    S_final = S_mvdr * gain
    
    # 8. Reconstruction
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    s_out /= np.max(np.abs(s_out))
    
    out_name = f"output_oracle_zoom_{USER_ZOOM_DEG}.wav"
    sf.write(f"{SAVE_DIR}/{out_name}", s_out, FS)
    print(f"Saved to {out_name}")

if __name__ == "__main__":
    main()