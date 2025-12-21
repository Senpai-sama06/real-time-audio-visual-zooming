#!/usr/bin/env python3
"""
unified_beamformers.py
MVDR and SMVB beamformers using shared noise covariance estimation.
Fair comparison with identical preprocessing.
"""

import numpy as np
import scipy.signal
import soundfile as sf
import os
from cov_estimator import NoiseCovarianceEstimator

# ------------------------
# Config / Constants
# ------------------------
FS = 16000
N_FFT = 256
N_HOP = 128
D = 0.08            # mic spacing
C = 343.0
ANGLE_TARGET = 90.0
SIGMA = 1e-3        # diagonal loading for MVDR
N_MICS = 2
SAVE_DIR = "sample"


# ========================================================
# STEERING VECTOR (Shared by both beamformers)
# ========================================================
def get_steering_vector(angle_deg, f, d, c):
    """Compute steering vector for target direction"""
    theta = np.deg2rad(angle_deg)
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f
    v = np.array([
        [np.exp(-1j * omega * tau1)],
        [np.exp(-1j * omega * tau2)]
    ], dtype=complex)
    return v


# ========================================================
# BEAMFORMER 1: MVDR (Minimum Variance Distortionless Response)
# ========================================================
def mvdr_beamformer(Y_mix, R_in, f_bins, angle_target=ANGLE_TARGET):
    """
    MVDR beamformer using pre-computed interference+noise covariance
    
    Args:
        Y_mix: Mixture STFT (n_channels, n_freqs, n_frames)
        R_in: Interference+noise covariance (n_freqs, n_channels, n_channels)
        f_bins: Frequency bins in Hz
        angle_target: Target angle in degrees
    
    Returns:
        S_mvdr: Beamformed output (n_freqs, n_frames)
    """
    n_channels, n_freqs, n_frames = Y_mix.shape
    S_mvdr = np.zeros((n_freqs, n_frames), dtype=complex)
    
    for f_idx in range(n_freqs):
        f_hz = f_bins[f_idx]
        
        # Low-freq bypass
        if f_hz < 100:
            S_mvdr[f_idx, :] = Y_mix[0, f_idx, :]
            continue
        
        # Get covariance with diagonal loading
        R = R_in[f_idx] + SIGMA * np.eye(n_channels)
        
        # Target steering vector
        d = get_steering_vector(angle_target, f_hz, D, C)
        
        # MVDR weight: w = R^{-1} * d / (d^H * R^{-1} * d)
        try:
            w = np.linalg.solve(R, d)
            w = w / (d.conj().T @ w + 1e-10)
        except np.linalg.LinAlgError:
            # Fallback to simple delay-and-sum
            w = np.array([[1.0], [0.0]], dtype=complex)
        
        # Apply beamformer
        S_mvdr[f_idx, :] = (w.conj().T @ Y_mix[:, f_idx, :]).squeeze()
    
    return S_mvdr


# ========================================================
# BEAMFORMER 2: SMVB (Hybrid Null Steering)
# ========================================================
def smvb_beamformer(Y_mix, R_in, f_bins, angle_target=ANGLE_TARGET):
    """
    SMVB (Spatial Matched-filter Voiceprint Beamformer) using null steering
    
    Args:
        Y_mix: Mixture STFT (n_channels, n_freqs, n_frames)
        R_in: Interference+noise covariance (n_freqs, n_channels, n_channels)
        f_bins: Frequency bins in Hz
        angle_target: Target angle in degrees
    
    Returns:
        S_smvb: Beamformed output (n_freqs, n_frames)
    """
    n_channels, n_freqs, n_frames = Y_mix.shape
    S_smvb = np.zeros((n_freqs, n_frames), dtype=complex)
    
    # Constraint vector for null steering
    desired = np.array([[1], [0]], dtype=np.complex64)
    
    for f_idx in range(n_freqs):
        f_hz = f_bins[f_idx]
        
        # Low-freq bypass
        if f_hz < 200:
            S_smvb[f_idx, :] = Y_mix[0, f_idx, :]
            continue
        
        # Get interference covariance
        R_int = R_in[f_idx]
        
        # Dominant interference eigenvector (largest eigenvalue)
        eigvals, eigvecs = np.linalg.eigh(R_int)
        v_int = eigvecs[:, -1].reshape(n_channels, 1)
        
        # Phase normalize interference vector
        v_int = v_int / (v_int[0] / (np.abs(v_int[0]) + 1e-10))
        
        # Target steering vector
        v_tgt = get_steering_vector(angle_target, f_hz, D, C)
        
        # Constraint matrix: [v_target | v_interference]
        C_mat = np.column_stack((v_tgt, v_int))
        
        # Check conditioning
        if np.linalg.cond(C_mat) > 10:
            # Poor conditioning, fallback to simple beamformer
            w = v_tgt / N_MICS
        else:
            # Solve constrained problem: C^H * w = [1; 0]
            try:
                w = np.linalg.solve(C_mat.conj().T, desired)
            except:
                w = v_tgt / N_MICS
        
        # Apply beamformer
        S_smvb[f_idx, :] = (w.conj().T @ Y_mix[:, f_idx, :]).squeeze()
    
    return S_smvb


# ========================================================
# POST-PROCESSING
# ========================================================
def apply_postfilter_and_istft(S_bf, mask_target, fs, n_fft, n_hop):
    """Apply spectral post-filter and convert back to time domain"""
    # Oracle spectral post-filter
    S_final = S_bf * mask_target
    
    # ISTFT
    _, s_out = scipy.signal.istft(S_final, fs=fs, nperseg=n_fft, noverlap=n_hop)
    
    # Normalize
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-10)
    
    return s_out


# ========================================================
# MAIN COMPARISON
# ========================================================
def main():
    print("="*70)
    print("UNIFIED BEAMFORMER COMPARISON: MVDR vs SMVB")
    print("Using shared noise covariance estimation from explicit noise.wav")
    print("="*70)
    
    # Step 1: Unified covariance estimation
    print("\n[1/4] Estimating noise covariance...")
    estimator = NoiseCovarianceEstimator(
        fs=FS,
        n_fft=N_FFT,
        n_hop=N_HOP,
        save_dir=SAVE_DIR
    )
    
    # Get all preprocessed data with sqrt weighting (recommended)
    results = estimator.estimate_full_pipeline(use_sqrt_weighting=True)
    
    Y_mix = results['Y_mix']
    R_in = results['R_in']
    mask_target = results['mask_target']
    f_bins = results['f_bins']
    
    # Step 2: MVDR Beamforming
    print("\n[2/4] Running MVDR beamformer...")
    S_mvdr = mvdr_beamformer(Y_mix, R_in, f_bins, ANGLE_TARGET)
    s_mvdr = apply_postfilter_and_istft(S_mvdr, mask_target, FS, N_FFT, N_HOP)
    
    out_mvdr = os.path.join(SAVE_DIR, "output_unified_mvdr.wav")
    sf.write(out_mvdr, s_mvdr.astype(np.float32), FS)
    print(f"✓ Saved: {out_mvdr}")
    
    # Step 3: SMVB Beamforming
    print("\n[3/4] Running SMVB beamformer...")
    S_smvb = smvb_beamformer(Y_mix, R_in, f_bins, ANGLE_TARGET)
    s_smvb = apply_postfilter_and_istft(S_smvb, mask_target, FS, N_FFT, N_HOP)
    
    out_smvb = os.path.join(SAVE_DIR, "output_unified_smvb.wav")
    sf.write(out_smvb, s_smvb.astype(np.float32), FS)
    print(f"✓ Saved: {out_smvb}")
    
    # Step 4: Summary
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print(f"Both beamformers used:")
    print(f"  • Identical noise covariance matrix (from explicit noise.wav)")
    print(f"  • Identical oracle masks")
    print(f"  • Identical pre/post-processing")
    print(f"\nKey difference:")
    print(f"  • MVDR: Minimizes output power (w^H * R * w) subject to d^H * w = 1")
    print(f"  • SMVB: Places null in interference direction via constrained optimization")
    print(f"\nOutputs saved for comparison:")
    print(f"  • {out_mvdr}")
    print(f"  • {out_smvb}")
    print("="*70)


if __name__ == "__main__":
    main()