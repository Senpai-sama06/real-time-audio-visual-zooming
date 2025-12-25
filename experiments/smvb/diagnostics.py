import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def run_gev_diagnostics(stft_multichannel, stft_target, stft_interferer, stft_noise, mask_speech, mask_noise, sample_rate=16000):
    """
    Runs comprehensive diagnostic tests for the GEV beamformer on simulated data.
    
    References for Metrics:
    - Eigenvalue Spread: Warsitz & Haeb-Umbach (Paper 2) [cite: 887, 888]
    - Distortion/Target Response: Warsitz & Haeb-Umbach [cite: 901]
    - BAN Gain Stability: Warsitz & Haeb-Umbach [cite: 921, 938]
    - Weight Norm (WNG): Nair et al. (Paper 1) [cite: 245]
    
    Args:
        stft_multichannel (np.ndarray): (F, T, M) Complex STFT of mixture.
        stft_target (np.ndarray): (F, T, M) Complex STFT of isolated target.
        stft_interferer (np.ndarray): (F, T, M) Complex STFT of isolated interferer.
        stft_noise (np.ndarray): (F, T, M) Complex STFT of isolated noise.
        mask_speech (np.ndarray): (F, T) Predicted speech mask.
        mask_noise (np.ndarray): (F, T) Predicted noise mask.
        sample_rate (int): Sampling rate for plotting axes.
        
    Returns:
        dict: A dictionary containing diagnostic metrics and arrays.
    """
    
    F, T, M = stft_multichannel.shape
    freqs = np.linspace(0, sample_rate/2, F)
    
    # Storage for diagnostics
    diag_metrics = {
        'eigenvalue_spread': np.zeros(F),
        'condition_number': np.zeros(F),
        'ban_gain': np.zeros(F),
        'weight_norm': np.zeros(F),
        'output_target': np.zeros((F, T), dtype=np.complex64),
        'output_interferer': np.zeros((F, T), dtype=np.complex64),
        'weights': np.zeros((F, M), dtype=np.complex64)
    }

    print("Running GEV Diagnostics...")
    
    # --- PHASE 1: Accumulate Statistics ---
    # Same logic as main implementation: creating Phi_XX and Phi_NN
    # Note: We use sqrt of mask for the weighted sum formulation
    X_weighted_speech = stft_multichannel * np.sqrt(mask_speech)[..., None]
    X_weighted_noise = stft_multichannel * np.sqrt(mask_noise)[..., None]
    
    # Compute Outer Products sum(mask * x * x^H)
    Phi_XX_all = np.einsum('ftm, ftn -> fmn', X_weighted_speech, X_weighted_speech.conj())
    Phi_NN_all = np.einsum('ftm, ftn -> fmn', X_weighted_noise, X_weighted_noise.conj())
    
    # Normalize by sum of weights
    weight_sum_s = np.sum(mask_speech, axis=1)[:, None, None] + 1e-10
    weight_sum_n = np.sum(mask_noise, axis=1)[:, None, None] + 1e-10
    
    Phi_XX_all /= weight_sum_s
    Phi_NN_all /= weight_sum_n
    
    # Diagonal Loading Factor (Regularization)
    DIAG_LOAD_FACTOR = 1e-6

    # --- PHASE 2: Per-Frequency Analysis ---
    for f in range(F):
        Phi_XX = Phi_XX_all[f]
        Phi_NN = Phi_NN_all[f]
        
        # TEST 5: Condition Number Test (Before Loading) [cite: 487]
        # Use singular values for condition number (svd is robust)
        s_vals = scipy.linalg.svdvals(Phi_NN)
        diag_metrics['condition_number'][f] = s_vals[0] / (s_vals[-1] + 1e-15)

        # Apply Loading (Crucial for stability)
        trace_val = np.trace(Phi_NN).real
        loading = (trace_val / M) * DIAG_LOAD_FACTOR
        Phi_NN_loaded = Phi_NN + np.eye(M) * loading
        
        # TEST 1: Eigenvalue Spread Test (Solver Health) 
        try:
            # eigh returns eigenvalues in ascending order
            eig_vals, eig_vecs = scipy.linalg.eigh(Phi_XX, Phi_NN_loaded)
            # Reverse to get descending order: lambda_1 >= lambda_2 >= ...
            eig_vals = eig_vals[::-1]
            eig_vecs = eig_vecs[:, ::-1]
            
            w_gev = eig_vecs[:, 0] # Principal eigenvector
            
            # Ratio of largest to second largest eigenvalue
            if len(eig_vals) > 1 and eig_vals[1] > 1e-10:
                diag_metrics['eigenvalue_spread'][f] = eig_vals[0] / eig_vals[1]
            else:
                diag_metrics['eigenvalue_spread'][f] = 1.0 # Confusion or single mic
                
        except np.linalg.LinAlgError:
            w_gev = np.ones(M) / np.sqrt(M)
            diag_metrics['eigenvalue_spread'][f] = 0.0

        # TEST 3: BAN Gain Stability [cite: 938]
        # Gain = sqrt(w^H * Phi * Phi * w / M) / (w^H * Phi * w)
        Rn_w = Phi_NN_loaded @ w_gev
        denom = np.real(w_gev.conj() @ Rn_w)
        num_term = np.real(Rn_w.conj() @ Rn_w)
        
        if denom > 1e-15:
            g_ban = np.sqrt(num_term / M) / denom
        else:
            g_ban = 0.0
        
        diag_metrics['ban_gain'][f] = g_ban
        
        # TEST 4: Weight Norm Test (White Noise Gain Proxy) [cite: 245]
        diag_metrics['weight_norm'][f] = np.real(np.linalg.norm(w_gev)**2)
        diag_metrics['weights'][f, :] = w_gev

        # TEST 2: Component-Wise Application (Transfer Function Check) [cite: 901]
        # Apply filters to ISOLATED components to see what happens to them
        # y = g * w^H * x
        
        # Target Response (Ref Mic 0 vs Output)
        target_f = stft_target[f] @ w_gev.conj()
        diag_metrics['output_target'][f, :] = target_f * g_ban
        
        # Interferer Response (Null Depth Check)
        interferer_f = stft_interferer[f] @ w_gev.conj()
        diag_metrics['output_interferer'][f, :] = interferer_f * g_ban

    # --- VISUALIZATION ---
    plot_diagnostics(diag_metrics, freqs, stft_target)
    
    return diag_metrics

def plot_diagnostics(metrics, freqs, stft_target_ref):
    """Generates plots for the diagnostics."""
    plt.figure(figsize=(15, 10))
    
    # 1. Eigenvalue Spread
    plt.subplot(2, 3, 1)
    plt.plot(freqs, 10 * np.log10(metrics['eigenvalue_spread'] + 1e-10))
    plt.title("Eigenvalue Spread (dB)")
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Ratio (dB)")
    plt.grid(True)
    
    # 2. BAN Gain Stability
    plt.subplot(2, 3, 2)
    plt.plot(freqs, metrics['ban_gain'])
    plt.title("BAN Post-Filter Gain")
    plt.xlabel("Freq (Hz)")
    plt.grid(True)
    
    # 3. Weight Norm (WNG)
    plt.subplot(2, 3, 3)
    plt.plot(freqs, 10 * np.log10(metrics['weight_norm'] + 1e-10))
    plt.title("Weight Norm (WNG Proxy)")
    plt.ylabel("dB")
    plt.xlabel("Freq (Hz)")
    plt.grid(True)
    
    # 4. Condition Number
    plt.subplot(2, 3, 4)
    plt.semilogy(freqs, metrics['condition_number'])
    plt.title("Condition Number of Phi_NN")
    plt.xlabel("Freq (Hz)")
    plt.grid(True)
    
    # 5. Spectral Distortion Check (Target Output vs Ref Mic 0)
    plt.subplot(2, 3, 5)
    # Average magnitude over time for comparison
    ref_spec = np.mean(np.abs(stft_target_ref[:, :, 0]), axis=1)
    out_spec = np.mean(np.abs(metrics['output_target']), axis=1)
    
    plt.plot(freqs, 20*np.log10(ref_spec + 1e-10), label='Ref Mic 0', alpha=0.5)
    plt.plot(freqs, 20*np.log10(out_spec + 1e-10), label='GEV Output', alpha=0.8)
    plt.title("Target Spectrum (Distortion Check)")
    plt.legend()
    plt.grid(True)
    
    # 6. Interferer Suppression
    plt.subplot(2, 3, 6)
    interf_out_spec = np.mean(np.abs(metrics['output_interferer']), axis=1)
    plt.plot(freqs, 20*np.log10(interf_out_spec + 1e-10), color='red')
    plt.title("Residual Interferer Level")
    plt.ylabel("dB")
    plt.xlabel("Freq (Hz)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()