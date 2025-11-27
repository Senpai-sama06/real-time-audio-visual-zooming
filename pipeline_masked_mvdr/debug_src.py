import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
import os
import sys

# --- Constants (Must match World.py) ---
FS = 16000
D = 0.01  # Changed from 0.04 to 0.01
C = 343.0

def get_steering_vector(angle_deg, f, d, c):
    theta_rad = np.deg2rad(angle_deg)
    tau_m1 = (d / 2) * np.cos(0) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(0) * np.cos(theta_rad - np.pi) / c
    omega = 2 * np.pi * f
    return np.array([np.exp(-1j * omega * tau_m1), np.exp(-1j * omega * tau_m2)])

def main(output_dir_world):
    # 1. Validation
    if not output_dir_world or not os.path.exists(output_dir_world):
        print(f"ERROR: Invalid directory provided: {output_dir_world}")
        return

    print(f"--- Debug: Analyzing data from '{output_dir_world}' ---")
    
    # 2. Find the Audio (It is inside the path passed to us)
    wav_path = os.path.join(output_dir_world, "mixture_3_sources.wav")
    
    if not os.path.exists(wav_path):
        print(f"ERROR: Audio file not found at: {wav_path}")
        return

    # --- PROCESSING (Standard SRP Logic) ---
    y, fs = sf.read(wav_path, dtype='float32')
    y = y.T
    
    f, t, Y_stft = scipy.signal.stft(y, fs=fs, nperseg=512, noverlap=256)
    n_freqs = len(f)
    angles = np.linspace(0, 180, 181)
    power_map = []
    
    print("Scanning angles...")
    for angle in angles:
        energy_at_angle = 0
        for i in range(n_freqs):
            if f[i] < 200 or f[i] > 4000: continue 
            d = get_steering_vector(angle, f[i], D, C)
            y_vec = Y_stft[:, i, :] 
            output = d.conj().T @ y_vec 
            energy_at_angle += np.sum(np.abs(output)**2)
        power_map.append(energy_at_angle)
        
    power_map = 10 * np.log10(power_map)
    power_map -= np.max(power_map) 

    # --- PLOTTING ---
    plt.figure(figsize=(10, 5))
    plt.plot(angles, power_map)
    plt.axvline(40, color='r', linestyle='--', label='True 40 (Int)')
    plt.axvline(90, color='g', linestyle='--', label='True 90 (Tgt)')
    plt.axvline(130, color='r', linestyle='--', label='True 130 (Int)')
    
    # Get the parent folder name (e.g. Simulation_Output_2025...) for the title
    parent_folder_name = os.path.basename(os.path.dirname(output_dir_world))
    plt.title(f"SRP Scan: {parent_folder_name}")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Energy (dB)")
    plt.legend()
    plt.grid(True)
    
    # --- NEW SAVING LOGIC ---
    
    simulation_root = os.path.dirname(output_dir_world)
    
    # 2. Create sibling folder "Debug_Outputs"
    debug_dir = os.path.join(simulation_root, "Debug_Outputs")
    os.makedirs(debug_dir, exist_ok=True)
    
    # 3. Save the plot
    plot_path = os.path.join(debug_dir, "debug_srp_plot.png")
    plt.savefig(plot_path)
    
    print(f"Done.")
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()