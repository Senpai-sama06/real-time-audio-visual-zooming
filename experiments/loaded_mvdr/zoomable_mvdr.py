import numpy as np
import scipy.signal
import soundfile as sf
import os
import matplotlib.pyplot as plt

# --- 1. Constants & THE "KNOB" ---
FS = 16000          # Sample rate (Must match world_builder.py)
D = 0.08            # Mic spacing (Must match world_builder.py)
C = 343.0           # Speed of sound
ANGLE_TARGET = 90.0 # Our look direction (Broadside)
N_MICS = 2

# STFT Parameters
N_PER_SEG = 512     # STFT window size (FFT size)
N_OVERLAP = 256     # 50% overlap

# --- YOUR "ZOOM" KNOB ---
# 1. "Zoom In" (Aggressive / Narrow Beam) -> SIGMA = 1e-9 
# 2. "Zoom Out" (Gentle / Wide Beam) -> SIGMA = 1.0 
#
SIGMA = 0  # <-- Set this to 1.0 or 1e-9 to test your zoom!


# --- 2. Core Physics Functions ---

def get_steering_vector(angle_deg, f, d, c, n_mics):
    """Calculates the 3D far-field steering vector for our array."""
    phi_rad = 0.0
    theta_rad = np.deg2rad(angle_deg)
    
    path_diff_m1 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - 0)
    path_diff_m2 = (d / 2) * np.cos(phi_rad) * np.cos(theta_rad - np.pi)

    tau_m1 = path_diff_m1 / c
    tau_m2 = path_diff_m2 / c
    
    omega = 2 * np.pi * f
    
    d_vec = np.array([
        [np.exp(-1j * omega * tau_m1)],
        [np.exp(-1j * omega * tau_m2)]
    ], dtype=complex)
    
    return d_vec

# --- UPDATED FUNCTION ---
def plot_beam_pattern(weights_per_freq, f_axis, d, c, n_mics, title_str, sigma_val):
    """
    Plots the final beam pattern and saves the plot with a
    sigma-specific filename.
    """
    angles_plot = np.linspace(0, 180, 180) 
    avg_response_db = np.zeros_like(angles_plot, dtype=float)
    
    for f_idx, f in enumerate(f_axis):
        if f == 0: continue
        w_f = weights_per_freq[f_idx] 
        
        response_f = []
        for angle in angles_plot:
            d_scan = get_steering_vector(angle, f, d, c, n_mics)
            gain = np.abs(w_f.conj().T @ d_scan)
            response_f.append(gain[0, 0])
            
        avg_response_db += 20 * np.log10(np.array(response_f))
        
    avg_response_db /= (len(f_axis) - 1)
        
    plt.figure(figsize=(10, 6))
    plt.plot(angles_plot, avg_response_db)
    # Use the passed sigma_val in the title
    plt.title(f"Average Beam Pattern: {title_str} (Sigma={sigma_val:.1e})")
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Average Gain (dB)')
    plt.ylim(-40, 5)
    plt.grid(True, which='both')
    
    plt.axvline(90, color='g', linestyle='--', label='S1 (90° Target)')
    plt.axvline(80, color='r', linestyle=':', label='S2 (40° Interferer)')
    plt.legend()
    
    # --- THIS IS THE CHANGE ---
    # Create a unique filename based on the sigma value
    output_plot_filename = f"beam_pattern_sigma_{sigma_val:.1e}.png"
    plt.savefig(output_plot_filename)
    print(f"Beam pattern saved to '{output_plot_filename}'")

# --- 3. Main Script ---

def main():
    print(f"--- 2. Zoomable MVDR: Processing Audio ---")
    print(f"--- Using 'Zoom' (sigma) = {SIGMA:.1e} ---")

    input_file = "mixture_2_sources.wav"
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        print("Please run the world_builder_2_source.py script first.")
        return
        
    y, fs = sf.read(input_file, dtype='float32')
    y = y.T 
    
    f, t, Y_stft = scipy.signal.stft(y, fs=fs, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
    n_channels, n_freqs, n_times = Y_stft.shape
    
    S_out_stft = np.zeros((n_freqs, n_times), dtype=complex)
    all_weights = np.zeros((n_freqs, n_channels, 1), dtype=complex)
    
    print("Processing frequency bins...")
    for f_idx in range(n_freqs):
        freq = f[f_idx]
        if freq == 0: continue 
            
        R = np.zeros((n_channels, n_channels), dtype=complex)
        for t_idx in range(n_times):
            Y_fk_t = Y_stft[:, f_idx, t_idx].reshape(n_channels, 1)
            R += Y_fk_t @ Y_fk_t.conj().T
        R /= n_times
        
        R_loaded = R + SIGMA * np.eye(n_channels)
        
        d_target = get_steering_vector(ANGLE_TARGET, freq, D, C, n_channels)
        
        try:
            R_inv_d = np.linalg.solve(R_loaded, d_target)
            denominator = d_target.conj().T @ R_inv_d
            if np.abs(denominator) < 1e-10: raise np.linalg.LinAlgError
            w = R_inv_d / denominator
        except np.linalg.LinAlgError:
            w = np.array([[1], [0]], dtype=complex)
        
        all_weights[f_idx] = w 
        
        S_out_stft[f_idx, :] = w.conj().T @ Y_stft[:, f_idx, :]

    print("Processing complete. Converting back to audio...")

    _, s_out = scipy.signal.istft(S_out_stft, fs=fs, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
    
    s_out = s_out[:y.shape[1]]
    
    max_val = np.max(np.abs(s_out))
    if max_val > 0:
        s_out /= max_val
    
    output_filename = f"output_2src_sigma_{SIGMA:.1e}.wav"
    sf.write(output_filename, s_out, fs)
    print(f"\nDone. Output saved to '{output_filename}'")
    
    # --- 4. Validation (UPDATED CALL) ---
    plot_title = "Zoom In (Narrow)" if SIGMA < 0.01 else "Zoom Out (Wide)"
    # Pass the SIGMA value to the plotting function
    plot_beam_pattern(all_weights, f, D, C, N_MICS, plot_title, SIGMA)

if __name__ == "__main__":
    main()