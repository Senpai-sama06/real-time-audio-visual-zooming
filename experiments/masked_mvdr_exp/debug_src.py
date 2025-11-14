import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt

# --- Constants (MUST MATCH WORLD.PY) ---
FS = 16000
D = 0.04
C = 343.0

def get_steering_vector(angle_deg, f, d, c):
    theta_rad = np.deg2rad(angle_deg)
    # Mic 1 (Right), Mic 2 (Left)
    tau_m1 = (d / 2) * np.cos(0) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(0) * np.cos(theta_rad - np.pi) / c
    omega = 2 * np.pi * f
    return np.array([np.exp(-1j * omega * tau_m1), np.exp(-1j * omega * tau_m2)])

def main():
    print("--- Debug: Steered Response Power (SRP) Scan ---")
    
    # Load Mixture
    y, fs = sf.read("mixture_3_sources.wav", dtype='float32')
    y = y.T
    
    # STFT
    f, t, Y_stft = scipy.signal.stft(y, fs=fs, nperseg=512, noverlap=256)
    n_freqs = len(f)
    
    # Scan Angles
    angles = np.linspace(0, 180, 181)
    power_map = []
    
    print("Scanning 0 to 180 degrees...")
    for angle in angles:
        # Calculate energy at this angle across all freqs
        energy_at_angle = 0
        
        for i in range(n_freqs):
            if f[i] < 200 or f[i] > 4000: continue # Look only in reliable band
            
            d = get_steering_vector(angle, f[i], D, C)
            
            # Simple Beamforming: w = d / M
            # Output = w.H * y
            y_vec = Y_stft[:, i, :] # [2, Time]
            
            # Beamform
            output = d.conj().T @ y_vec # [1, Time]
            
            # Power
            energy_at_angle += np.sum(np.abs(output)**2)
            
        power_map.append(energy_at_angle)
        
    # Plot
    power_map = 10 * np.log10(power_map)
    power_map -= np.max(power_map) # Normalize
    
    plt.figure(figsize=(10, 5))
    plt.plot(angles, power_map)
    plt.axvline(40, color='r', linestyle='--', label='True 40 (Int)')
    plt.axvline(90, color='g', linestyle='--', label='True 90 (Tgt)')
    plt.axvline(130, color='r', linestyle='--', label='True 130 (Int)')
    plt.title("Diagnostic: Does the Math match the Audio?")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Energy (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("debug_srp_plot.png")
    print("Saved 'debug_srp_plot.png'. Check the peaks!")

if __name__ == "__main__":
    main()