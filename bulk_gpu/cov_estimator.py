#!/usr/bin/env python3
"""
noise_covariance_estimator.py
Unified noise covariance estimation module for both MVDR and SMVB beamformers.
Uses explicit noise.wav file for consistent, fair comparison.
"""

import numpy as np
import soundfile as sf
import scipy.signal
import os


class NoiseCovarianceEstimator:
    """
    Estimates noise and interference covariance matrices using oracle masks
    constructed from explicit reference signals (target.wav, interference.wav, noise.wav)
    """
    
    def __init__(self, fs=16000, n_fft=256, n_hop=128, save_dir="sample"):
        self.fs = fs
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.save_dir = save_dir
        
    def load_signals(self):
        """Load all reference signals and mixture"""
        mix_path = os.path.join(self.save_dir, "mixture.wav")
        tgt_path = os.path.join(self.save_dir, "target.wav")
        int_path = os.path.join(self.save_dir, "interference.wav")
        noise_path = os.path.join(self.save_dir, "noise.wav")
        
        # Check files exist
        for p in (mix_path, tgt_path, int_path, noise_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")
        
        # Load mixture (stereo)
        y_mix, sr = sf.read(mix_path, dtype="float32")
        if sr != self.fs:
            print(f"Warning: mixture SR {sr} != FS {self.fs}")
        if y_mix.ndim == 1:
            raise ValueError("mixture.wav must be stereo (2-channel)")
        
        # Load references (stereo -> mono, take channel 0)
        s_t_st, _ = sf.read(tgt_path, dtype="float32")
        s_i_st, _ = sf.read(int_path, dtype="float32")
        s_n_st, _ = sf.read(noise_path, dtype="float32")
        
        s_t = s_t_st[:, 0] if s_t_st.ndim > 1 else s_t_st
        s_i = s_i_st[:, 0] if s_i_st.ndim > 1 else s_i_st
        s_n = s_n_st[:, 0] if s_n_st.ndim > 1 else s_n_st
        
        return y_mix.T, s_t, s_i, s_n  # Return mixture as (2, samples)
    
    def compute_stfts(self, y_mix, s_t, s_i, s_n):
        """Compute STFTs for all signals"""
        f_bins, t_bins, Y_mix = scipy.signal.stft(
            y_mix, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_hop
        )
        _, _, S_t = scipy.signal.stft(s_t, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_hop)
        _, _, S_i = scipy.signal.stft(s_i, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_hop)
        _, _, S_n = scipy.signal.stft(s_n, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_hop)
        
        return f_bins, t_bins, Y_mix, S_t, S_i, S_n
    
    def compute_oracle_masks(self, S_t, S_i, S_n):
        """
        Compute noise-aware oracle masks using explicit noise reference
        
        Returns:
            mask_target: Target mask (for spatial post-filtering)
            mask_interference_noise: Interference+noise mask (for covariance estimation)
        """
        mag_t = np.abs(S_t)
        mag_i = np.abs(S_i)
        mag_n = np.abs(S_n)
        
        eps = 1e-10
        
        # Target mask (power domain)
        mask_target = (mag_t ** 2) / (mag_t ** 2 + mag_i ** 2 + mag_n ** 2 + eps)
        
        # Complementary mask for interference+noise
        mask_interference_noise = 1.0 - mask_target
        
        return mask_target, mask_interference_noise
    
    def estimate_interference_noise_covariance(self, Y_mix, mask_int_noise, 
                                               use_sqrt_weighting=True):
        """
        Estimate interference+noise covariance matrix using masked observations
        
        Args:
            Y_mix: Mixture STFT (n_channels, n_freqs, n_frames)
            mask_int_noise: Interference+noise mask (n_freqs, n_frames)
            use_sqrt_weighting: If True, use sqrt of mask (theoretically correct for spatial cov)
        
        Returns:
            R_in: Interference+noise covariance (n_freqs, n_channels, n_channels)
        """
        n_channels, n_freqs, n_frames = Y_mix.shape
        R_in = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
        
        for f_idx in range(n_freqs):
            # Weight selection
            if use_sqrt_weighting:
                w = np.sqrt(mask_int_noise[f_idx, :])  # Square root for spatial covariance
            else:
                w = mask_int_noise[f_idx, :]  # Direct mask
            
            Yf = Y_mix[:, f_idx, :]  # (n_channels, n_frames)
            Yw = Yf * w  # Weighted observations
            
            # Normalized covariance
            denom = np.sum(w ** 2) + 1e-8
            R_in[f_idx] = (Yw @ Yw.conj().T) / denom
        
        return R_in
    
    def estimate_full_pipeline(self, use_sqrt_weighting=True):
        """
        Complete pipeline: load signals, compute masks, estimate covariance
        
        Returns:
            Dictionary containing all computed quantities for use by beamformers
        """
        # Load signals
        y_mix, s_t, s_i, s_n = self.load_signals()
        
        # Compute STFTs
        f_bins, t_bins, Y_mix, S_t, S_i, S_n = self.compute_stfts(y_mix, s_t, s_i, s_n)
        
        # Compute oracle masks
        mask_target, mask_int_noise = self.compute_oracle_masks(S_t, S_i, S_n)
        
        # Estimate covariance
        R_in = self.estimate_interference_noise_covariance(
            Y_mix, mask_int_noise, use_sqrt_weighting
        )
        
        print(f"✓ Loaded signals and computed STFTs")
        print(f"✓ Computed noise-aware oracle masks")
        print(f"✓ Estimated interference+noise covariance (sqrt_weighting={use_sqrt_weighting})")
        print(f"  - Shape: {R_in.shape}")
        print(f"  - Frequency bins: {len(f_bins)}")
        print(f"  - Time frames: {len(t_bins)}")
        
        return {
            'f_bins': f_bins,
            't_bins': t_bins,
            'Y_mix': Y_mix,
            'S_t': S_t,
            'S_i': S_i,
            'S_n': S_n,
            'mask_target': mask_target,
            'mask_int_noise': mask_int_noise,
            'R_in': R_in,
            'fs': self.fs,
            'n_fft': self.n_fft,
            'n_hop': self.n_hop
        }


# ========================================================
# EXAMPLE USAGE
# ========================================================
if __name__ == "__main__":
    # Initialize estimator
    estimator = NoiseCovarianceEstimator(
        fs=16000,
        n_fft=256,
        n_hop=128,
        save_dir="sample"
    )
    
    # Run full pipeline with sqrt weighting (recommended for spatial covariance)
    results = estimator.estimate_full_pipeline(use_sqrt_weighting=True)
    
    # Access computed quantities
    R_in = results['R_in']
    mask_target = results['mask_target']
    Y_mix = results['Y_mix']
    
    print("\n" + "="*60)
    print("Covariance estimation complete!")
    print("="*60)
    print(f"R_in can now be used by both MVDR and SMVB beamformers")
    print(f"mask_target can be used for spectral post-filtering")