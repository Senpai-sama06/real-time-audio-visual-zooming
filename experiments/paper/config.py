from dataclasses import dataclass
import numpy as np

@dataclass
class SystemConfig:
    """
    Central Configuration for Spatio-Spectral Hypothesis Testing.
    Groups parameters by Physical, Statistical, and Geometric domains.
    """
    
    # --- 1. PHYSICAL CONSTANTS ---
    fs: int = 16000          # Sampling Rate (Hz)
    n_fft: int = 512         # FFT Window Size
    mic_dist: float = 0.08   # Distance between microphones (meters)
    c: float = 343.0         # Speed of Sound (m/s)
    target_angle: float = 1.5708 # Look direction (radians, 0=Broadside)

    # --- 2. STATISTICAL ESTIMATION (Phase 1) ---
    # Covariance Smoothing (approx 100ms integration time)
    alpha: float = 0.85      
    
    # Numerical Stability (Diagonal Loading)
    diagonal_load: float = 1e-6 
    
    # Noise Floor Tracking (Jeub/Nelke)
    jeub_decay: float = 0.999   # Slow decay (forgetting factor)
    jeub_attack: float = 0.1    # Fast attack for minima update
    
    # MDL Rank Criterion
    # Effective number of snapshots = 1 / (1 - alpha)
    mdl_snapshots: float = 10.0
    
    # Energy Gating
    snr_thresh_db: float = 3.0  # Minimum SNR to attempt classification

    # --- 3. GEOMETRIC DECISION BOUNDARIES (Phase 2) ---
    # Coherence Reliability
    coherence_gate: float = 0.3 # Minimum |Gamma| to trust phase info
    
    # Schwarz Line Test
    # Normalized distance threshold (variance-weighted)
    line_thresh: float = 0.15   
    
    # Diffuseness Threshold (for H4 vs H6)
    diffuse_thresh: float = 0.3 # If t < 0.3, it's mostly diffuse