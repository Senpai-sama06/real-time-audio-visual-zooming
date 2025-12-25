import numpy as np
import scipy.linalg

def gev(stft_multichannel, mask_speech):
    """
    Implements Offline GEV Beamforming with BAN Post-Filtering.
    Assumes Noise Mask = 1.0 - Speech Mask.
    
    References:
    - GEV Solver: Warsitz & Haeb-Umbach (Paper 2), Eq. (5)
    - BAN Post-Filter: Warsitz & Haeb-Umbach (Paper 2), Eq. (17)
    - Diagonal Loading: Warsitz & Haeb-Umbach (Paper 2), Section V.A
    
    Args:
        stft_multichannel (np.ndarray): Shape (F, T, M). Complex STFT of M mics.
        mask_speech (np.ndarray): Shape (F, T). Predicted speech mask (0.0 to 1.0).
        
    Returns:
        stft_enhanced (np.ndarray): Shape (F, T). Enhanced single-channel STFT.
    """
    
    # --- Setup ---
    F, T, M = stft_multichannel.shape
    stft_enhanced = np.zeros((F, T), dtype=np.complex64)
    
    # Derive Noise Mask (Simplified Logic)
    mask_noise = 1.0 - mask_speech
    
    # Diagonal Loading Factor: Ensures Phi_NN is invertible.
    # The paper suggests -30dB regularization.
    # We use a relative factor of 1e-6 (approx -60dB) which is usually stable enough.
    DIAG_LOAD_FACTOR = 1e-6

    print(f"Starting GEV Beamforming (F={F}, T={T}, M={M})...")

    # --- PHASE 1: Covariance Accumulation (Global Statistics) ---
    # We compute the weighted sum of outer products over all time T.
    
    # Prepare data for broadcasting: (F, T, M) * (F, T, 1)
    X_weighted_speech = stft_multichannel * np.sqrt(mask_speech)[..., None]
    X_weighted_noise = stft_multichannel * np.sqrt(mask_noise)[..., None]

    # Compute Global Covariance Matrices (Phi_XX and Phi_NN)
    # einsum 'ftm, ftn -> fmn' performs the outer product (x @ x.H) and sums over T
    Phi_XX_all = np.einsum('ftm, ftn -> fmn', X_weighted_speech, X_weighted_speech.conj())
    Phi_NN_all = np.einsum('ftm, ftn -> fmn', X_weighted_noise, X_weighted_noise.conj())

    # Normalize by total mask weight to get the true average
    weight_sum_s = np.sum(mask_speech, axis=1)[:, None, None] + 1e-10
    weight_sum_n = np.sum(mask_noise, axis=1)[:, None, None] + 1e-10
    
    Phi_XX_all /= weight_sum_s
    Phi_NN_all /= weight_sum_n

    # --- PHASE 2, 3 & 4: Processing per Frequency Bin ---
    
    for f in range(F):
        # Extract matrices for this frequency
        Phi_XX = Phi_XX_all[f]
        Phi_NN = Phi_NN_all[f]
        
        # Apply Diagonal Loading (Regularization) to Noise Matrix
        # Formula: Phi_NN + (trace(Phi_NN)/M * factor) * I
        trace_val = np.trace(Phi_NN).real
        loading = (trace_val / M) * DIAG_LOAD_FACTOR
        Phi_NN += np.eye(M) * loading

        # -- SOLVER (GEV) --
        # Solve generalized eigenvalue problem: Phi_XX * w = lambda * Phi_NN * w
        try:
            # eigh returns eigenvalues in ascending order
            eig_vals, eig_vecs = scipy.linalg.eigh(Phi_XX, Phi_NN)
            # Select eigenvector corresponding to the LARGEST eigenvalue 
            w_gev = eig_vecs[:, -1]
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular (rare with loading)
            w_gev = np.ones(M) / np.sqrt(M) 

        # -- CORRECTOR (BAN Post-Filter) --
        # Calculate Blind Analytical Normalization gain 
        # Formula: numerator = sqrt( (w^H * Phi_NN * Phi_NN * w) / M )
        #          denominator = w^H * Phi_NN * w
        
        # Helper: vector-matrix product (Phi_NN * w)
        Rn_w = Phi_NN @ w_gev
        
        # Denominator term (w^H * Rn * w) - result is real scalar
        denom = np.real(w_gev.conj() @ Rn_w)
        
        # Numerator term (w^H * Rn * Rn * w)
        # Note: w^H * Rn * Rn * w is same as (Rn*w)^H * (Rn*w) -> norm squared of Rn_w vector
        num_term = np.real(Rn_w.conj() @ Rn_w)
        numerator = np.sqrt(num_term / M)
        
        # Compute Gain (prevent divide by zero)
        if denom > 1e-15:
            g_ban = numerator / denom
        else:
            g_ban = 1.0

        # -- RECONSTRUCTION --
        # Apply weights: y(t) = g * w^H * x(t)
        # stft_multichannel[f] is (T, M)
        # w_gev.conj() is (M,)
        
        beamformed_f = stft_multichannel[f] @ w_gev.conj()
        stft_enhanced[f, :] = beamformed_f * g_ban

    print("GEV Processing Complete.")
    return stft_enhanced