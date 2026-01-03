import numpy as np
from config import SystemConfig
from anchors import PhysicsAnchors

def detect_aliasing(features, anchors):
    """Level 0: Checks if frequency bin is spatially aliased."""
    decisions = np.zeros(len(features.gamma), dtype=bool)
    decisions[anchors.alias_bin_idx:] = True
    return decisions

def detect_silence(features, config: SystemConfig):
    """Level 1: Energy Gating based on SNR."""
    return features.snr_db < config.snr_thresh_db

def detect_rank_1_wax_mdl(features, config: SystemConfig):
    """
    Level 1: Wax & Kailath MDL Criterion for M=2.
    Returns Boolean Mask: True if Rank 1 (Signal), False if Rank 2 (Full/Noise).
    """
    l1 = features.evals[:, 0]
    l2 = features.evals[:, 1]
    N = config.mdl_snapshots
    
    # MDL(k) = -log(L) + 0.5 * k(2M-k) * log(N)
    
    # MDL for k=0 (Noise Only)
    # log(L) term depends on GMEAN/AMEAN ratio
    geo = np.sqrt(l1 * l2)
    ari = (l1 + l2) / 2 + 1e-12
    # DoF = 0
    mdl_0 = -N * 2 * np.log(geo/ari + 1e-12) 
    
    # MDL for k=1 (One Source)
    # Remaining eigenvalues = [l2]. Ratio = 1. LogL = 0.
    # DoF = 1 * (4 - 1) = 3
    mdl_1 = 0.5 * 3 * np.log(N)
    
    # Decision: Is MDL_1 < MDL_0?
    # We add a small bias to favor Rank 2 in ambiguity (conservative)
    return mdl_1 < mdl_0

def analyze_rank_1_direction(features, anchors, config: SystemConfig):
    """
    Level 2 Branch A: Checks if Rank 1 signal aligns with Target or Jammer.
    Returns: Array of codes (1=Target, 5=Jammer, 0=Uncertain)
    """
    codes = np.zeros(len(features.gamma), dtype=int)
    
    # Phase Difference
    tgt_phase = np.angle(anchors.target)
    obs_phase = np.angle(features.gamma)
    
    # Circular Distance
    err = np.abs(np.angle(np.exp(1j * (obs_phase - tgt_phase))))
    
    # Reliability Gate (Phase 2.1)
    is_reliable = np.abs(features.gamma) > config.coherence_gate
    
    # Classification
    # If reliable and close -> Target (1)
    # If reliable and far -> Jammer (5)
    # If unreliable -> Treat as Diffuse/Noise (0 -> will fallback)
    
    mask_tgt = (err < np.pi/3) & is_reliable
    mask_jam = (err >= np.pi/3) & is_reliable
    
    codes[mask_tgt] = 1
    codes[mask_jam] = 5
    
    return codes


def analyze_rank_2_geometry(features, anchors, config: SystemConfig):
    codes = np.zeros(len(features.gamma), dtype=int)
    
    # ... [Keep existing Anchors setup A, B, P] ...
    A = anchors.diffuse
    B = anchors.target
    P = features.gamma
    
    # --- INSERT THIS BLOCK ---
    # Shortcut: If Coherence is effectively zero, it is Diffuse (H6)
    # This prevents geometric instability near the origin
    mag_gamma = np.abs(features.gamma)
    is_zero_coh = mag_gamma < 0.3
    
    # ... [Keep Projection Logic] ...
    AB = B - A
    AP = P - A
    t = np.real(AP * np.conj(AB)) / (np.abs(AB)**2 + 1e-12)
    t_clamped = np.clip(t, 0, 1)
    
    Closest = A + t_clamped * AB
    
    # ... [Keep Variance Normalization] ...
    dist = np.abs(P - Closest)
    mag_sq = mag_gamma**2
    var_proxy = (1 - mag_sq) + 0.1 
    norm_dist = dist / var_proxy
    
    # ... [Update Decisions Logic] ...
    is_off_line = norm_dist > config.line_thresh
    is_diffuse = (t_clamped < config.diffuse_thresh) | is_zero_coh # <--- Add OR condition
    
    codes[is_off_line] = 2
    codes[~is_off_line & is_diffuse] = 6
    codes[~is_off_line & ~is_diffuse] = 4
    
    return codes