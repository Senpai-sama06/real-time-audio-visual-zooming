import numpy as np
from config import SystemConfig
from anchors import PhysicsAnchors
from features import FeatureExtractor
from temporal import TemporalSmoother
import detectors

class SpatioSpectralClassifier:
    def __init__(self, target_angle_rad=1.57):
        # 1. Load Configuration
        self.cfg = SystemConfig(target_angle=target_angle_rad)
        
        # 2. Precompute Physics
        self.anchors = PhysicsAnchors(self.cfg)
        
        # 3. Initialize Stateful Features
        self.features = FeatureExtractor(self.cfg)
        
        # 4. Initialize Temporal Smoothing
        self.smoother = TemporalSmoother(self.features.n_bins, window_size=5)

    def process_frame(self, Y_frame):
        """
        Main Pipeline.
        Y_frame: [Bins, 2] complex input
        Returns: [Bins] integer hypothesis codes
        """
        # --- Step A: Update State (Covariance, Noise, SNR) ---
        feats = self.features.update(Y_frame)
        
        # --- Step B: Level 0 & 1 Detectors (Stateless) ---
        mask_aliasing = detectors.detect_aliasing(feats, self.anchors)
        mask_silence = detectors.detect_silence(feats, self.cfg)
        mask_rank1 = detectors.detect_rank_1_wax_mdl(feats, self.cfg)
        
        # --- Step C: Level 2 Detectors (Geometry) ---
        # Branch A: Rank 1 Directional Check
        codes_rank1 = detectors.analyze_rank_1_direction(feats, self.anchors, self.cfg)
        
        # Branch B: Rank 2 Geometric Mixture Check
        codes_rank2 = detectors.analyze_rank_2_geometry(feats, self.anchors, self.cfg)
        
        # --- Step D: Fusion Logic ---
        final_decisions = np.zeros(self.features.n_bins, dtype=int)
        
        for f in range(self.features.n_bins):
            # Priority 1: Physics Limit
            if mask_aliasing[f]:
                final_decisions[f] = 7
                continue
                
            # Priority 2: Energy Gate
            if mask_silence[f]:
                final_decisions[f] = 0
                continue
                
            # Priority 3: Rank Logic
            if mask_rank1[f]:
                # It's Rank 1. Is it Target(1) or Jammer(5)?
                # Note: codes_rank1 handles reliability gating internally.
                # If unreliable, it returns 0. Here we fallback to Diffuse(6) if unreliable Rank 1.
                code = codes_rank1[f]
                final_decisions[f] = code if code != 0 else 6 
            else:
                # It's Rank 2. Use Geometry.
                final_decisions[f] = codes_rank2[f]

        # --- Step E: Temporal Smoothing ---
        smoothed_decisions = self.smoother.update(final_decisions)
        
        return smoothed_decisions