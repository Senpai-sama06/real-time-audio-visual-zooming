import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import stft, istft
from scipy.io import wavfile
import os
import sys

# Import system components
from config import SystemConfig
from classifier import SpatioSpectralClassifier
from actuator import SpatioSpectralActuator

# --- 1. DATA LOADER ---
def load_world_files(base_path="sample"):
    files = {
        "mix": os.path.join(base_path, "mixture.wav"),
        "tgt": os.path.join(base_path, "target.wav"),
        "int": os.path.join(base_path, "interference.wav"),
        "noi": os.path.join(base_path, "noise.wav")
    }
    data = {}
    fs = 16000
    for key, path in files.items():
        if not os.path.exists(path):
            print(f"Error: Missing {path}."); sys.exit(1)
        fs_in, audio = wavfile.read(path)
        # Normalize
        if audio.dtype == np.int16: audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32: audio = audio.astype(np.float32) / 2147483648.0
        else: audio = audio.astype(np.float32)
        data[key] = audio
        fs = fs_in
    return fs, data

def get_oracle_labels(P_tgt, P_int, P_noi):
    labels = np.zeros_like(P_tgt, dtype=int)
    thresh = 1.0 
    mask_tgt = (P_tgt > (P_int + P_noi) * thresh)
    mask_int = (P_int > (P_tgt + P_noi) * thresh)
    labels[:, :] = 6 
    labels[mask_int] = 5 
    labels[mask_tgt] = 1 
    return labels

# --- 2. ACTUATOR ---
class DiagnosticActuator(SpatioSpectralActuator):
    def get_weights(self, Y_frame, R_nn, state):
        weights = np.zeros((self.n_bins, 2), dtype=complex)
        for f in range(self.n_bins):
            s = state[f]
            if s in [0, 3, 7]: weights[f, 0] = 1.0 
            else:
                if s in [2, 5]: mu = 0.01; Phi = R_nn[f]
                elif s in [4, 6]: mu = 1.0; Phi = self.Phi_diffuse[f] * np.real(R_nn[f,0,0])
                else: mu = 10.0; Phi = np.eye(2) * 1e-6
                
                d = np.array([1.0, self.d_vec[f]], dtype=complex)
                Phi_reg = Phi + np.eye(2) * self.cfg.diagonal_load
                try:
                    x = np.linalg.solve(Phi_reg, d)
                    w = x / (np.vdot(d, x).real + 1e-12)
                    if s in [5, 6]: w *= 0.1 
                    weights[f] = w
                except: weights[f] = np.array([0.5, 0.5])
        return weights

class MetricTracker:
    def __init__(self, name):
        self.name, self.p_tgt_in, self.p_int_in, self.p_tgt_out, self.p_int_out = name, 0., 0., 0., 0.
    def update(self, w, Y_tgt, Y_int):
        S_tgt = np.sum(w.conj() * Y_tgt, axis=1)
        S_int = np.sum(w.conj() * Y_int, axis=1)
        self.p_tgt_in += np.sum(np.abs(Y_tgt[:,0])**2)
        self.p_int_in += np.sum(np.abs(Y_int[:,0])**2)
        self.p_tgt_out += np.sum(np.abs(S_tgt)**2)
        self.p_int_out += np.sum(np.abs(S_int)**2)
    def report(self):
        eps = 1e-9
        return (10*np.log10(self.p_tgt_out/(self.p_tgt_in+eps)), 
                10*np.log10(self.p_int_out/(self.p_int_in+eps)),
                10*np.log10(self.p_tgt_in/(self.p_int_in+eps)), 
                10*np.log10(self.p_tgt_out/(self.p_int_out+eps)))

if __name__ == "__main__":
    # --- 1. FORCE CONFIGURATION ---
    # We manually override the config to ensure correct physics
    cfg = SystemConfig(mic_dist=0.08, target_angle=1.5708)
    
    print(f"Diagnostics Configured for: Angle={np.rad2deg(cfg.target_angle):.1f} deg | Dist={cfg.mic_dist} m")
    
    # --- 2. INITIALIZE CLASSIFIER ---
    # Passing the angle explicitly
    classifier = SpatioSpectralClassifier(target_angle_rad=cfg.target_angle)
    
    # --- 3. THE "BRAIN SURGERY" CHECK ---
    # We inspect the internal Anchor Phase for Bin 100 (~3kHz)
    # For Broadside (90 deg), Phase should be 0.0
    # For Endfire (0 deg), Phase would be large (~2.0 rad)
    test_bin = 100
    anchor_phase = np.angle(classifier.anchors.target[test_bin])
    
    print("\n" + "="*40)
    print(f"INTERNAL CHECK (Bin {test_bin}):")
    print(f"  > Expected Phase for Broadside: 0.00")
    print(f"  > Classifier Internal Phase:    {anchor_phase:.4f}")
    
    if abs(anchor_phase) > 0.1:
        print("  [FAIL] CLASSIFIER IS STILL USING ENDFIRE (0 deg)!")
        print("         Action: Hard-code '1.5708' inside classifier.py __init__ as default.")
        sys.exit(1)
    else:
        print("  [PASS] Classifier is correctly locked to Broadside.")
    print("="*40 + "\n")

    # --- 4. RUN PIPELINE ---
    print("Loading Data...")
    fs, audio = load_world_files()
    n_fft = cfg.n_fft
    
    def get_stft(x): 
        _, _, Zxx = stft(x[:,0], fs=fs, nperseg=n_fft)
        _, _, Zyy = stft(x[:,1], fs=fs, nperseg=n_fft)
        return np.stack([Zxx, Zyy], axis=-1)

    Y_mix = get_stft(audio['mix'])
    Y_tgt = get_stft(audio['tgt'])
    Y_int = get_stft(audio['int'])
    Y_oracle_noise = get_stft(audio['int'] + audio['noi']) 
    
    P_tgt = np.abs(Y_tgt[:,:,0])**2 + np.abs(Y_tgt[:,:,1])**2
    P_int = np.abs(Y_int[:,:,0])**2 + np.abs(Y_int[:,:,1])**2
    
    print("Running Classification...")
    oracle_map = get_oracle_labels(P_tgt, P_int, np.zeros_like(P_tgt))
    est_map = np.zeros_like(oracle_map)
    for i in range(Y_mix.shape[1]):
        est_map[:, i] = classifier.process_frame(Y_mix[:, i, :])
        
    print("Running Actuation...")
    actuator = DiagnosticActuator(cfg) # Use same config!
    m_oracle = MetricTracker("Oracle"); m_real = MetricTracker("Real System")
    S_oracle = np.zeros(Y_mix.shape[0:2], dtype=complex)
    S_real = np.zeros(Y_mix.shape[0:2], dtype=complex)
    
    for i in range(Y_mix.shape[1]):
        y_m, y_t, y_i = Y_mix[:, i, :], Y_tgt[:, i, :], Y_int[:, i, :]
        y_on = Y_oracle_noise[:, i, :] 
        
        # Oracle
        R_oracle = y_on[:, :, np.newaxis] @ y_on[:, :, np.newaxis].conj().transpose(0, 2, 1)
        w_o = actuator.get_weights(y_m, R_oracle, oracle_map[:, i])
        S_oracle[:, i] = np.sum(w_o.conj() * y_m, axis=1)
        m_oracle.update(w_o, y_t, y_i)
        
        # Real
        w_r = actuator.get_weights(y_m, classifier.features.R_smooth, est_map[:, i])
        S_real[:, i] = np.sum(w_r.conj() * y_m, axis=1)
        m_real.update(w_r, y_t, y_i)

    op_tgt, op_jam, op_sinr_in, op_sinr_out = m_oracle.report()
    rp_tgt, rp_jam, rp_sinr_in, rp_sinr_out = m_real.report()
    
    mask = (oracle_map == 1) | (oracle_map == 5)
    acc = np.mean(est_map[mask] == oracle_map[mask]) * 100 if np.sum(mask) > 0 else 0

    print("\n" + "="*65)
    print(f"{' COMPREHENSIVE DIAGNOSTIC REPORT ':^65}")
    print("="*65)
    print(f"1. ESTIMATOR ACCURACY: {acc:.1f}%")
    print("-" * 65)
    print(f"{'METRIC':<20} | {'ORACLE':<15} | {'REAL':<15}")
    print("-" * 65)
    print(f"{'Output SINR':<20} | {op_sinr_out:>6.2f} dB        | {rp_sinr_out:>6.2f} dB")
    print(f"{'Net SINR Gain':<20} | {op_sinr_out-op_sinr_in:>6.2f} dB        | {rp_sinr_out-rp_sinr_in:>6.2f} dB")
    print("-" * 65)
    
    print("Generating Plots...")
    t_axis = np.linspace(0, len(audio['mix'])/fs, Y_mix.shape[1])
    f_axis = np.linspace(0, fs/2, Y_mix.shape[0])
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    cmap = mcolors.ListedColormap(['black','lime','red','yellow','cyan','orange','magenta','white'])
    
    ax[0,0].imshow(oracle_map, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=7)
    ax[0,0].set_title("Ground Truth")
    ax[0,1].imshow(est_map, aspect='auto', origin='lower', cmap=cmap, vmin=0, vmax=7)
    ax[0,1].set_title(f"Estimator (Acc: {acc:.1f}%)")
    
    ax[1,0].pcolormesh(t_axis, f_axis, 10*np.log10(np.abs(S_oracle)**2 + 1e-12), cmap='inferno')
    ax[1,0].set_title(f"Oracle Output ({op_sinr_out:.1f}dB)")
    ax[1,1].pcolormesh(t_axis, f_axis, 10*np.log10(np.abs(S_real)**2 + 1e-12), cmap='inferno')
    ax[1,1].set_title(f"Real Output ({rp_sinr_out:.1f}dB)")
    
    plt.tight_layout()
    plt.show()
    
    _, wav_r = istft(S_real, fs=fs, nperseg=n_fft)
    mx = np.max(np.abs(wav_r)); wav_r = wav_r/mx if mx > 0 else wav_r
    wavfile.write("diag_real.wav", fs, wav_r.astype(np.float32))