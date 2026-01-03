import numpy as np
from dataclasses import dataclass
from config import SystemConfig

@dataclass
class SpatialFeatures:
    """
    Container for extracted features per frequency bin.
    """
    R: np.ndarray          # Spatial Covariance Matrix [Bins, 2, 2]
    evals: np.ndarray      # Eigenvalues [Bins, 2]
    gamma: np.ndarray      # Complex Coherence [Bins]
    noise_floor: np.ndarray # Estimated Noise PSD [Bins]
    snr_db: np.ndarray     # Estimated SNR [Bins]

class FeatureExtractor:
    """
    Stateful processor that maintains recursive estimates of:
    1. Spatial Covariance Matrix (R_yy)
    2. Noise Floor (Minima Tracking)
    """
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self.n_bins = config.n_fft // 2 + 1
        
        # Phase 1.1: Robust Initialization (Scaled Identity)
        self.R_smooth = np.zeros((self.n_bins, 2, 2), dtype=complex)
        for f in range(self.n_bins):
            self.R_smooth[f] = np.eye(2) * self.cfg.diagonal_load
            
        # Phase 1.2: Noise Floor Init (High value to allow drop)
        # made change here
        self.noise_floor = np.ones(self.n_bins) * 1e-6

    def update(self, Y_frame) -> SpatialFeatures:
        """
        Updates internal state with new frame Y [Bins, 2]
        Returns extracted features.
        """
        # --- 1. Covariance Update ---
        # Outer product: Y * Y^H -> [Bins, 2, 2]
        Y = Y_frame[:, :, np.newaxis]
        R_inst = Y @ Y.conj().transpose(0, 2, 1)
        
        # Recursive Smoothing
        self.R_smooth = self.cfg.alpha * self.R_smooth + (1 - self.cfg.alpha) * R_inst
        
        # Diagonal Loading (Numerical Stability)
        # Add epsilon to diagonal elements only
        idx = np.arange(self.n_bins)
        self.R_smooth[idx, 0, 0] += self.cfg.diagonal_load * 1e-4
        self.R_smooth[idx, 1, 1] += self.cfg.diagonal_load * 1e-4

        # --- 2. Eigenvalue Decomposition (Closed Form for 2x2) ---
        # Trace and Determinant
        tr = np.trace(self.R_smooth, axis1=1, axis2=2).real
        det = np.linalg.det(self.R_smooth).real
        
        # L1,2 = Tr/2 +/- sqrt((Tr/2)^2 - Det)
        delta = np.sqrt(np.maximum((tr/2)**2 - det, 0))
        l1 = tr/2 + delta
        l2 = tr/2 - delta
        
        # Safety: l2 can be slightly negative due to float precision, clamp it
        l2 = np.maximum(l2, 1e-12)
        evals = np.stack([l1, l2], axis=1)

        # --- 3. Noise Floor Tracking (Phase 1.2 Fix) ---
        # Track minima of the Trace (Total Power)
        # Logic: Fast Attack (drop), Slow Decay (rise)
        current_energy = tr
        
        # Update rule: 
        # If Energy < Floor: Update fast (0.1)
        # If Energy > Floor: Update slow (0.001) -> Drift up
        
        mask_min = current_energy < self.noise_floor
        
        # Fast Drop
        self.noise_floor[mask_min] = (self.cfg.jeub_attack * self.noise_floor[mask_min] + 
                                      (1 - self.cfg.jeub_attack) * current_energy[mask_min])
        
        # Slow Rise (Simulating forgetting factor)
        self.noise_floor[~mask_min] *= 1.00005 # Approx mult by 1.0001

        # --- 4. Coherence & SNR ---
        # Gamma = R_01 / sqrt(R_00 * R_11)
        num = self.R_smooth[:, 0, 1]
        den = np.sqrt(self.R_smooth[:,0,0].real * self.R_smooth[:,1,1].real + 1e-12)
        gamma = num / den
        
        snr_db = 10 * np.log10(current_energy / (self.noise_floor + 1e-9))

        return SpatialFeatures(
            R=self.R_smooth,
            evals=evals,
            gamma=gamma,
            noise_floor=self.noise_floor,
            snr_db=snr_db
        )