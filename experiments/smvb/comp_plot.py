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
INPUT_SNR_RANGE = range(-10, 31, 5)  
FS = 16000
FFT_SIZE = 512
HOP_SIZE = 256

# --- [CONTROL KNOB] MASK IMPERFECTION ---
# 0.0 = Perfect Oracle (Upper Bound)
# 0.2 = Good Neural Network (Realistic)
# 0.5 = Poor Neural Network
# 1.0 = Random Noise
MASK_ERROR_ALPHA = 1 

# --- 2. METRIC FUNCTIONS ---

def compute_sisdr(reference, estimate):
    """Scale-Invariant Signal-to-Distortion Ratio."""
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    dot_product = np.sum(reference * estimate)
    ref_energy = np.sum(reference ** 2) + 1e-10
    alpha = dot_product / ref_energy
    target_part = alpha * reference
    noise_part = estimate - target_part
    sisdr = 10 * np.log10(np.sum(target_part**2) / (np.sum(noise_part**2) + 1e-10))
    return sisdr

def compute_output_snr(reference, estimate):
    """Output SNR (Signal vs Residual)."""
    dot = np.sum(reference * estimate)
    ref_en = np.sum(reference**2) + 1e-10
    scale = dot / ref_en
    error = estimate - (scale * reference)
    p_sig = np.sum((scale * reference)**2)
    p_err = np.sum(error**2) + 1e-10
    return 10 * np.log10(p_sig / p_err)

# --- 3. MASK UTILS ---

def degrade_mask(clean_mask, alpha):
    """
    Mixes the clean mask with uniform noise to simulate estimation errors.
    """
    if alpha <= 0:
        return clean_mask
    F, T = clean_mask.shape
    random_mask = np.random.rand(F, T)
    # Linear mix: (1-alpha)*Clean + alpha*Random
    noisy_mask = (1 - alpha) * clean_mask + alpha * random_mask
    return noisy_mask

def compute_oracle_masks_from_files(target_wav, interf_wav, noise_wav):
    target_stft = stft(target_wav) 
    noise_complex_wav = interf_wav + noise_wav
    noise_stft = stft(noise_complex_wav)
    P_target = np.mean(np.abs(target_stft)**2, axis=2)
    P_noise = np.mean(np.abs(noise_stft)**2, axis=2)
    mask_speech = P_target / (P_target + P_noise + 1e-12)
    return mask_speech

# --- 4. BEAMFORMER FUNCTIONS ---

def gev(stft_multichannel, mask_speech):
    """Offline GEV Beamforming with BAN Post-Filtering."""
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
    """Hard-Nulling (MVDR) Beamformer."""
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

# --- 5. UTILS ---

def stft(x):
    x = x.T 
    f, t, Zxx = scipy.signal.stft(x, fs=FS, nperseg=FFT_SIZE, noverlap=FFT_SIZE-HOP_SIZE)
    return Zxx.transpose(1, 2, 0)

def istft(X):
    t, x = scipy.signal.istft(X, fs=FS, nperseg=FFT_SIZE, noverlap=FFT_SIZE-HOP_SIZE)
    return x

# --- 6. REPORTING ---

def write_comprehensive_report(results, filename="experiment_report.txt"):
    print(f"Generating report: {filename}...")
    with open(filename, "w") as f:
        f.write("========================================================\n")
        f.write("   BEAMFORMER COMPARISON STUDY: GEV vs HARD-NULLING     \n")
        f.write("========================================================\n\n")
        f.write(f"Mask Imperfection (Alpha): {MASK_ERROR_ALPHA} (0=Perfect, 1=Random)\n")
        f.write(f"SNR Range: {min(INPUT_SNR_RANGE)}dB to {max(INPUT_SNR_RANGE)}dB\n")
        
        for metric in ['pesq', 'sisdr', 'sinr']:
            f.write(f"\n--- METRIC: {metric.upper()} ---\n")
            for n_interf in NUM_INTERFERERS_RANGE:
                gev_scores = np.array(results['gev'][n_interf][metric])
                hn_scores = np.array(results['hn'][n_interf][metric])
                avg_gev = np.mean(gev_scores)
                avg_hn = np.mean(hn_scores)
                diff = avg_gev - avg_hn
                winner = "GEV" if diff > 0 else "HN"
                
                f.write(f"[{n_interf} Interf] Avg: GEV={avg_gev:.2f} | HN={avg_hn:.2f} -> Winner: {winner}\n")
    print("Report complete.")

# --- 7. PLOTTING ---

def plot_all_metrics(results):
    metrics = ['pesq', 'sisdr', 'sinr']
    titles = ['PESQ', 'Si-SDR', 'Output SNR']
    ylims = [(1.0, 4.6), (-5, 25), (-5, 25)]
    
    for metric, title, ylim in zip(metrics, titles, ylims):
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()
        
        for i, n_interf in enumerate(NUM_INTERFERERS_RANGE):
            ax = axes[i]
            gev_data = results['gev'][n_interf][metric]
            hn_data = results['hn'][n_interf][metric]
            
            ax.plot(INPUT_SNR_RANGE, gev_data, 'b-o', label='GEV', linewidth=2)
            ax.plot(INPUT_SNR_RANGE, hn_data, 'r-s', label='HN', linewidth=2, linestyle='--')
            
            ax.set_title(f"{n_interf} Interferer(s)")
            ax.set_xlabel("Input SINR (dB)")
            ax.set_ylabel(metric.upper())
            ax.set_ylim(ylim) 
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            
        axes[7].axis('off')
        plt.suptitle(f"{title} (Mask Error Alpha={MASK_ERROR_ALPHA})", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"results_{metric}_alpha{MASK_ERROR_ALPHA}.png")
        plt.close()

# --- 8. MAIN LOOP ---

def main():
    results = {
        'gev': {n: {'pesq': [], 'sisdr': [], 'sinr': []} for n in NUM_INTERFERERS_RANGE},
        'hn': {n: {'pesq': [], 'sisdr': [], 'sinr': []} for n in NUM_INTERFERERS_RANGE}
    }
    
    print("==================================================")
    print(f"   STUDY START | Mask Imperfection Alpha: {MASK_ERROR_ALPHA}   ")
    print("==================================================")
    
    for n_interf in NUM_INTERFERERS_RANGE:
        print(f"\n[Scenario] Simulating {n_interf} Interferer(s)...")
        subprocess.run(["python", "world.py", "--n", str(n_interf), "--reverb"], check=True)
        
        # Load & Truncate
        target_wav, _ = sf.read("sample/target.wav")
        interf_wav, _ = sf.read("sample/interference.wav")
        noise_wav, _ = sf.read("sample/noise.wav") 
        min_len = min(len(target_wav), len(interf_wav), len(noise_wav))
        target_wav, interf_wav, noise_wav = target_wav[:min_len], interf_wav[:min_len], noise_wav[:min_len]
        
        print(f"  > Sweeping SNR...")
        for target_snr in INPUT_SNR_RANGE:
            # Mix
            noise_complex = interf_wav + noise_wav
            p_t = np.mean(target_wav**2)
            p_n = np.mean(noise_complex**2)
            scalar = 0 if p_n == 0 else np.sqrt(p_t / (p_n * 10**(target_snr/10)))
            scaled_noise_complex = noise_complex * scalar
            mixture = target_wav + scaled_noise_complex
            
            # Masks
            clean_mask_s = compute_oracle_masks_from_files(target_wav, interf_wav*scalar, noise_wav*scalar)
            
            # --- APPLY CONTROLLED IMPERFECTION ---
            mask_s = degrade_mask(clean_mask_s, MASK_ERROR_ALPHA)
            mask_n = 1.0 - mask_s
            
            # Process
            mix_stft = stft(mixture)
            out_gev = istft(gev(mix_stft, mask_s))
            out_hn = istft(run_hn_lcmv(mix_stft, mask_n))
            
            # Eval
            ref = target_wav[:, 0]
            L = min(len(ref), len(out_gev), len(out_hn))
            ref, out_gev, out_hn = ref[:L], out_gev[:L], out_hn[:L]
            
            # Metrics
            sisdr_g = compute_sisdr(ref, out_gev)
            sisdr_h = compute_sisdr(ref, out_hn)
            sinr_g = compute_output_snr(ref, out_gev)
            sinr_h = compute_output_snr(ref, out_hn)
            
            try:
                pesq_g = pesq(FS, ref/(np.max(np.abs(ref))+1e-9), out_gev/(np.max(np.abs(out_gev))+1e-9), 'wb')
                pesq_h = pesq(FS, ref/(np.max(np.abs(ref))+1e-9), out_hn/(np.max(np.abs(out_hn))+1e-9), 'wb')
            except:
                pesq_g, pesq_h = 1.0, 1.0
            
            results['gev'][n_interf]['pesq'].append(pesq_g)
            results['gev'][n_interf]['sisdr'].append(sisdr_g)
            results['gev'][n_interf]['sinr'].append(sinr_g)
            
            results['hn'][n_interf]['pesq'].append(pesq_h)
            results['hn'][n_interf]['sisdr'].append(sisdr_h)
            results['hn'][n_interf]['sinr'].append(sinr_h)
            
            print(f"    SNR: {target_snr}dB | PESQ: {pesq_g:.2f}/{pesq_h:.2f}")

    plot_all_metrics(results)
    write_comprehensive_report(results)

if __name__ == "__main__":
    main()