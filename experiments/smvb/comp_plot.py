import numpy as np
import scipy.linalg
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
import subprocess
import os
from pesq import pesq  # pip install pesq

# --- 1. CONFIGURATION ---
NUM_INTERFERERS_RANGE = range(1, 8)  
INPUT_SNR_RANGE = range(-10, 31, 1)  
FS = 16000
FFT_SIZE = 512
HOP_SIZE = 256

# --- 2. METRIC FUNCTIONS (NEW) ---

def compute_sisdr(reference, estimate):
    """
    Scale-Invariant Signal-to-Distortion Ratio (Si-SDR).
    Measures separation quality independent of gain.
    """
    # Ensure zero mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Calculate optimal scaling factor (alpha)
    dot_product = np.sum(reference * estimate)
    ref_energy = np.sum(reference ** 2) + 1e-10
    alpha = dot_product / ref_energy
    
    # Projection
    target_part = alpha * reference
    noise_part = estimate - target_part
    
    # Energy Ratios
    e_target = np.sum(target_part ** 2) + 1e-10
    e_noise = np.sum(noise_part ** 2) + 1e-10
    
    sisdr = 10 * np.log10(e_target / e_noise)
    return sisdr

def compute_output_snr(reference, estimate):
    """
    Standard Output SNR (Signal vs Residual).
    """
    # Align energy (simple scaling to minimize MSE)
    # This approximates the 'true' signal component in the output
    dot = np.sum(reference * estimate)
    ref_en = np.sum(reference**2) + 1e-10
    scale = dot / ref_en
    
    error = estimate - (scale * reference)
    
    p_sig = np.sum((scale * reference)**2)
    p_err = np.sum(error**2) + 1e-10
    
    return 10 * np.log10(p_sig / p_err)

# --- 3. BEAMFORMER FUNCTIONS ---

def gev(stft_multichannel, mask_speech):
    """
    Implements Offline GEV Beamforming with BAN Post-Filtering.
    Assumes Noise Mask = 1.0 - Speech Mask.
    """
    F, T, M = stft_multichannel.shape
    stft_enhanced = np.zeros((F, T), dtype=np.complex64)
    mask_noise = 1.0 - mask_speech
    DIAG_LOAD_FACTOR = 1e-6

    X_weighted_speech = stft_multichannel * np.sqrt(mask_speech)[..., None]
    X_weighted_noise = stft_multichannel * np.sqrt(mask_noise)[..., None]

    Phi_XX_all = np.einsum('ftm, ftn -> fmn', X_weighted_speech, X_weighted_speech.conj())
    Phi_NN_all = np.einsum('ftm, ftn -> fmn', X_weighted_noise, X_weighted_noise.conj())

    weight_sum_s = np.sum(mask_speech, axis=1)[:, None, None] + 1e-10
    weight_sum_n = np.sum(mask_noise, axis=1)[:, None, None] + 1e-10
    
    Phi_XX_all /= weight_sum_s
    Phi_NN_all /= weight_sum_n

    for f in range(F):
        Phi_XX = Phi_XX_all[f]
        Phi_NN = Phi_NN_all[f]
        
        trace_val = np.trace(Phi_NN).real
        loading = (trace_val / M) * DIAG_LOAD_FACTOR
        Phi_NN += np.eye(M) * loading

        try:
            eig_vals, eig_vecs = scipy.linalg.eigh(Phi_XX, Phi_NN)
            w_gev = eig_vecs[:, -1]
        except np.linalg.LinAlgError:
            w_gev = np.ones(M) / np.sqrt(M) 

        Rn_w = Phi_NN @ w_gev
        denom = np.real(w_gev.conj() @ Rn_w)
        num_term = np.real(Rn_w.conj() @ Rn_w)
        numerator = np.sqrt(num_term / M)
        
        if denom > 1e-15:
            g_ban = numerator / denom
        else:
            g_ban = 1.0

        beamformed_f = stft_multichannel[f] @ w_gev.conj()
        stft_enhanced[f, :] = beamformed_f * g_ban
        
    return stft_enhanced

def run_hn_lcmv(stft_multichannel, mask_noise):
    """
    Hard-Nulling (MVDR) Beamformer.
    """
    F, T, M = stft_multichannel.shape
    stft_enhanced = np.zeros((F, T), dtype=np.complex64)
    DIAG_LOAD_FACTOR = 1e-3 
    
    X_weighted_noise = stft_multichannel * np.sqrt(mask_noise)[..., None]
    Phi_NN_all = np.einsum('ftm, ftn -> fmn', X_weighted_noise, X_weighted_noise.conj())
    
    weight_sum_n = np.sum(mask_noise, axis=1)[:, None, None] + 1e-10
    Phi_NN_all /= weight_sum_n
    
    v = np.ones(M, dtype=np.complex64)
    
    for f in range(F):
        Phi_NN = Phi_NN_all[f]
        loading = np.trace(Phi_NN.real) / M * DIAG_LOAD_FACTOR
        Phi_NN += np.eye(M) * loading
        
        try:
            Phi_inv_v = scipy.linalg.solve(Phi_NN, v, assume_a='pos')
            denom = v.conj() @ Phi_inv_v
            if denom < 1e-15:
                w = np.ones(M)/M
            else:
                w = Phi_inv_v / denom
        except np.linalg.LinAlgError:
            w = np.ones(M)/M
            
        stft_enhanced[f, :] = stft_multichannel[f] @ w.conj()
        
    return stft_enhanced

# --- 4. UTILS ---

def stft(x):
    x = x.T 
    f, t, Zxx = scipy.signal.stft(x, fs=FS, nperseg=FFT_SIZE, noverlap=FFT_SIZE-HOP_SIZE)
    return Zxx.transpose(1, 2, 0)

def istft(X):
    t, x = scipy.signal.istft(X, fs=FS, nperseg=FFT_SIZE, noverlap=FFT_SIZE-HOP_SIZE)
    return x

def compute_oracle_masks_from_files(target_wav, interf_wav, noise_wav):
    target_stft = stft(target_wav) 
    noise_complex_wav = interf_wav + noise_wav
    noise_stft = stft(noise_complex_wav)
    P_target = np.mean(np.abs(target_stft)**2, axis=2)
    P_noise = np.mean(np.abs(noise_stft)**2, axis=2)
    mask_speech = P_target / (P_target + P_noise + 1e-12)
    return mask_speech

# --- 5. REPORTING FUNCTION (NEW) ---

def write_comprehensive_report(results, filename="experiment_report.txt"):
    """
    Writes a detailed text summary of the comparison.
    """
    print(f"Generating summary report: {filename}...")
    
    with open(filename, "w") as f:
        f.write("========================================================\n")
        f.write("   BEAMFORMER COMPARISON STUDY: GEV vs HARD-NULLING     \n")
        f.write("========================================================\n\n")
        f.write(f"Metrics Evaluated: PESQ, Si-SDR, Output SNR\n")
        f.write(f"SNR Range: {min(INPUT_SNR_RANGE)}dB to {max(INPUT_SNR_RANGE)}dB\n")
        f.write(f"Interferers Tested: {min(NUM_INTERFERERS_RANGE)} to {max(NUM_INTERFERERS_RANGE)}\n\n")

        for metric in ['pesq', 'sisdr', 'sinr']:
            f.write(f"--------------------------------------------------------\n")
            f.write(f"   ANALYSIS FOR METRIC: {metric.upper()}\n")
            f.write(f"--------------------------------------------------------\n")
            
            for n_interf in NUM_INTERFERERS_RANGE:
                gev_scores = np.array(results['gev'][n_interf][metric])
                hn_scores = np.array(results['hn'][n_interf][metric])
                
                avg_gev = np.mean(gev_scores)
                avg_hn = np.mean(hn_scores)
                diff = avg_gev - avg_hn
                
                winner = "GEV" if diff > 0 else "HN"
                
                f.write(f"\n[Scenario: {n_interf} Interferers]\n")
                f.write(f"  > Average {metric.upper()}: GEV={avg_gev:.2f} | HN={avg_hn:.2f}\n")
                f.write(f"  > Winner: {winner} (by {abs(diff):.2f})\n")
                
                # Check low SNR performance (-10dB)
                idx_low = 0 # -10dB is first index
                f.write(f"  > At -10dB SNR: GEV={gev_scores[idx_low]:.2f} vs HN={hn_scores[idx_low]:.2f}\n")
                
                # Check high SNR performance (30dB)
                idx_high = -1 # 30dB is last index
                f.write(f"  > At +30dB SNR: GEV={gev_scores[idx_high]:.2f} vs HN={hn_scores[idx_high]:.2f}\n")
            
            f.write("\n")
            
    print("Report generation complete.")

# --- 6. PLOTTING FUNCTION (UPDATED) ---

def plot_all_metrics(results):
    metrics = ['pesq', 'sisdr', 'sinr']
    titles = ['PESQ (Perceptual Quality)', 'Si-SDR (Separation Quality)', 'Output SNR (Noise Reduction)']
    ylims = [(1.0, 4.6), (-5, 25), (-5, 25)]
    
    for metric, title, ylim in zip(metrics, titles, ylims):
        print(f"Plotting {metric}...")
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()
        
        for i, n_interf in enumerate(NUM_INTERFERERS_RANGE):
            ax = axes[i]
            gev_data = results['gev'][n_interf][metric]
            hn_data = results['hn'][n_interf][metric]
            
            ax.plot(INPUT_SNR_RANGE, gev_data, 'b-o', label='GEV', linewidth=2)
            ax.plot(INPUT_SNR_RANGE, hn_data, 'r-s', label='HN', linewidth=2, linestyle='--')
            
            ax.set_title(f"{n_interf} Interferer(s)", fontweight='bold')
            ax.set_xlabel("Input SINR (dB)")
            ax.set_ylabel(metric.upper())
            ax.set_ylim(ylim) 
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right')
            
        axes[7].axis('off')
        plt.suptitle(f"Beamformer Comparison: {title}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"results_{metric}.png")
        plt.close() # Close plot to free memory

# --- 7. MAIN STUDY LOOP ---

def main():
    # Structure: results['gev'][n_interf]['pesq'] = [scores...]
    results = {
        'gev': {n: {'pesq': [], 'sisdr': [], 'sinr': []} for n in NUM_INTERFERERS_RANGE},
        'hn': {n: {'pesq': [], 'sisdr': [], 'sinr': []} for n in NUM_INTERFERERS_RANGE}
    }
    
    print("==================================================")
    print("   STUDY: GEV vs HN (Oracle Clean-File Masks)     ")
    print("==================================================")
    
    for n_interf in NUM_INTERFERERS_RANGE:
        print(f"\n[Scenario] Running World Sim for {n_interf} Interferer(s)...")
        subprocess.run(["python", "world.py", "--n", str(n_interf), "--reverb"], check=True)
        
        # Load Raw Components (Stereo)
        target_wav, _ = sf.read("sample/target.wav")
        interf_wav, _ = sf.read("sample/interference.wav")
        noise_wav, _ = sf.read("sample/noise.wav") 
        
        min_len = min(len(target_wav), len(interf_wav), len(noise_wav))
        target_wav = target_wav[:min_len]
        interf_wav = interf_wav[:min_len]
        noise_wav = noise_wav[:min_len]
        
        print(f"  > Sweeping SNR for N={n_interf}...")

        for target_snr in INPUT_SNR_RANGE:
            # --- DYNAMIC MIXING ---
            noise_complex = interf_wav + noise_wav
            
            p_target = np.mean(target_wav**2)
            p_noise = np.mean(noise_complex**2)
            
            if p_noise == 0: scalar = 0
            else:
                req_ratio = 10**(target_snr / 10.0)
                scalar = np.sqrt(p_target / (p_noise * req_ratio))
            
            scaled_interf = interf_wav * scalar
            scaled_noise = noise_wav * scalar
            mixture = target_wav + scaled_interf + scaled_noise
            
            # --- PROCESSING ---
            m_speech = compute_oracle_masks_from_files(target_wav, scaled_interf, scaled_noise)
            m_noise = 1.0 - m_speech
            
            mix_stft = stft(mixture)
            out_gev = istft(gev(mix_stft, m_speech))
            out_hn = istft(run_hn_lcmv(mix_stft, m_noise))
            
            # --- EVALUATION ---
            ref_sig = target_wav[:, 0]
            L = min(len(ref_sig), len(out_gev), len(out_hn))
            
            # Truncate
            ref_sig = ref_sig[:L]
            out_gev = out_gev[:L]
            out_hn = out_hn[:L]
            
            # 1. Si-SDR
            sisdr_gev = compute_sisdr(ref_sig, out_gev)
            sisdr_hn = compute_sisdr(ref_sig, out_hn)
            
            # 2. Output SNR
            sinr_gev = compute_output_snr(ref_sig, out_gev)
            sinr_hn = compute_output_snr(ref_sig, out_hn)
            
            # 3. PESQ (Normalized)
            ref_n = ref_sig / (np.max(np.abs(ref_sig)) + 1e-9)
            gev_n = out_gev / (np.max(np.abs(out_gev)) + 1e-9)
            hn_n = out_hn / (np.max(np.abs(out_hn)) + 1e-9)
            
            try:
                pesq_gev = pesq(FS, ref_n, gev_n, 'wb')
                pesq_hn = pesq(FS, ref_n, hn_n, 'wb')
            except:
                pesq_gev = 1.0
                pesq_hn = 1.0
            
            # Store
            results['gev'][n_interf]['pesq'].append(pesq_gev)
            results['gev'][n_interf]['sisdr'].append(sisdr_gev)
            results['gev'][n_interf]['sinr'].append(sinr_gev)
            
            results['hn'][n_interf]['pesq'].append(pesq_hn)
            results['hn'][n_interf]['sisdr'].append(sisdr_hn)
            results['hn'][n_interf]['sinr'].append(sinr_hn)
            
            print(f"    SNR: {target_snr}dB | PESQ: {pesq_gev:.2f} | SiSDR: {sisdr_gev:.1f}")

    # --- SAVE RESULTS ---
    plot_all_metrics(results)
    write_comprehensive_report(results)

if __name__ == "__main__":
    main()