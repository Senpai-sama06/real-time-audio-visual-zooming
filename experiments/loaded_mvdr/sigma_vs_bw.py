import numpy as np
import scipy.signal
import soundfile as sf
import os
import matplotlib.pyplot as plt

# --- 1. Constants ---
FS = 16000
D = 0.04
C = 343.0
ANGLE_TARGET = 90.0
ANGLE_INTERFERER = 40.0 # <-- The angle we will measure
N_MICS = 2
N_PER_SEG = 512
N_OVERLAP = 256

# --- YOUR "KNOB" - 1000 POINTS ---
# We will test 1000 points on a log scale, from 1e-10 to 10
SIGMA_LIST = np.logspace(-10, 1, 1000)

# --- 2. Core Physics Functions (Unchanged) ---

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

# --- 3. NEW METRIC: NULL DEPTH CALCULATOR ---

def get_beam_pattern_data(weights_per_freq, f_axis, d, c, n_mics, target_angle=90.0):
    """
    Calculates the beam pattern data, returns the arrays.
    Normalized to the target angle.
    """
    angles_plot = np.linspace(0, 180, 361) # 0.5 degree resolution
    avg_response_linear = np.zeros_like(angles_plot, dtype=float)
    
    num_freqs = 0
    for f_idx, f in enumerate(f_axis):
        if f <= 100: continue
        w_f = weights_per_freq[f_idx]
        
        response_f_linear = []
        for angle in angles_plot:
            d_scan = get_steering_vector(angle, f, d, c, n_mics)
            gain_linear = np.abs(w_f.conj().T @ d_scan)[0, 0]
            response_f_linear.append(gain_linear**2) # Average power
            
        avg_response_linear += np.array(response_f_linear)
        num_freqs += 1
        
    avg_response_linear /= num_freqs
    
    avg_response_linear[avg_response_linear < 1e-20] = 1e-20
    avg_response_db = 10 * np.log10(avg_response_linear)
    
    # Normalize the peak *at 90 deg* to be 0dB
    target_idx = np.argmin(np.abs(angles_plot - target_angle))
    gain_at_target = avg_response_db[target_idx]
    avg_response_db -= gain_at_target 
    
    return angles_plot, avg_response_db

def get_gain_at_angle(angles_plot, avg_response_db, query_angle):
    """
    Finds the gain (in dB) at a specific angle.
    """
    query_idx = np.argmin(np.abs(angles_plot - query_angle))
    gain_at_query = avg_response_db[query_idx]
    return gain_at_query

# --- 5. Main Script ---

def main():
    print("--- 4. MVDR 'Aggressiveness' vs. Sigma Experiment ---")
    print(f"--- (Running {len(SIGMA_LIST)} sigma points) ---")

    input_file = "mixture_2_sources.wav"
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return
        
    y, fs = sf.read(input_file, dtype='float32')
    y = y.T 
    
    f, t, Y_stft = scipy.signal.stft(y, fs=fs, nperseg=N_PER_SEG, noverlap=N_OVERLAP)
    n_channels, n_freqs, n_times = Y_stft.shape
    
    print("Pre-calculating covariance matrix...")
    R_per_freq = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for f_idx in range(n_freqs):
        if f[f_idx] <= 100: continue
        R_f = np.zeros((n_channels, n_channels), dtype=complex)
        for t_idx in range(n_times):
            Y_fk_t = Y_stft[:, f_idx, t_idx].reshape(n_channels, 1)
            R_f += Y_fk_t @ Y_fk_t.conj().T
        R_per_freq[f_idx] = R_f / n_times

    # --- This is your experiment ---
    results_sigma = []
    results_null_depth = []

    print("Running experiment for each SIGMA value...")
    
    for i, SIGMA in enumerate(SIGMA_LIST):
        
        all_weights = np.zeros((n_freqs, n_channels, 1), dtype=complex)
        
        for f_idx in range(n_freqs):
            freq = f[f_idx]
            if freq <= 100: continue 
                
            R = R_per_freq[f_idx]
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
        
        # --- 6. Calculate and Store Results ---
        angles_plot, avg_response_db = get_beam_pattern_data(all_weights, f, D, C, N_MICS, ANGLE_TARGET)
        
        # --- THIS IS OUR NEW METRIC ---
        # Get the gain at the 40-degree interferer location
        null_depth_at_40 = get_gain_at_angle(angles_plot, avg_response_db, ANGLE_INTERFERER)
        
        results_sigma.append(SIGMA)
        results_null_depth.append(null_depth_at_40)
        
        if i % 100 == 0: # Print progress
            print(f"  > Progress: {i/len(SIGMA_LIST)*100:.0f}%... (SIGMA={SIGMA:.1e}, Null Depth={null_depth_at_40:.1f} dB)")

    print(f"  > Progress: 100%... Done.")

    # --- 7. Plot the Final Relationship ---
    print("\nPlotting final relationship...")
    plt.figure(figsize=(10, 6))
    plt.plot(results_sigma, results_null_depth) # Note: 'o-' is too slow for 1000 points
    plt.xscale('log') 
    plt.xlabel('SIGMA Value (Diagonal Loading) - "Gentleness"')
    plt.ylabel(f'Null Depth at {ANGLE_INTERFERER}° (dB)')
    plt.title(f'MVDR "Aggressiveness": Null Depth vs. Sigma')
    plt.grid(True, which='both')
    plt.savefig("sigma_vs_null_depth.png")
    
    print("Done. Final plot saved to 'sigma_vs_null_depth.png'")
    plt.show()


if __name__ == "__main__":
    main()