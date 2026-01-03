import numpy as np
from config import SystemConfig
from anchors import PhysicsAnchors

class SpatioSpectralActuator:
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self.n_bins = config.n_fft // 2 + 1
        
        # Precompute Steering Vector for Look Direction (0 deg)
        self.anchors = PhysicsAnchors(config)
        self.d_vec = self.anchors.target # [Bins] complex phase shift
        
        # Precompute Diffuse Coherence Matrix (Phi_diffuse)
        # Shape: [Bins, 2, 2]
        self.Phi_diffuse = np.zeros((self.n_bins, 2, 2), dtype=complex)
        for f in range(self.n_bins):
            # Diagonals are 1, Off-diagonals are Sinc
            coh = self.anchors.diffuse[f]
            self.Phi_diffuse[f] = np.array([[1, coh], [coh, 1]])

        # TSNR State (Smoothed SNR for stability)
        self.xi_prio = np.zeros(self.n_bins) # A priori SNR
        self.G_prev = np.ones(self.n_bins)   # Previous Gain

    def process_frame(self, Y_frame, R_nn, decisions):
        """
        Inputs:
            Y_frame:   [Bins, 2] complex STFT
            R_nn:      [Bins, 2, 2] Noise Covariance (from Estimator/Features)
            decisions: [Bins] Integer Hypothesis codes (0-7)
        Returns:
            S_hat:     [Bins] complex enhanced spectrum (Single Channel output)
        """
        S_hat = np.zeros(self.n_bins, dtype=complex)
        
        # Loop over frequencies (Vectorized where possible, but logic splits per bin)
        for f in range(self.n_bins):
            state = decisions[f]
            y_t = Y_frame[f, :] # [2x1]
            
            # --- PATH A: PHYSICS FAILURES (Spectral Processing) ---
            # H7 (Aliasing), H3 (Collinear), H0 (Silence)
            if state == 7 or state == 3 or state == 0:
                S_hat[f] = self._path_a_spectral(f, y_t, R_nn[f], state)
                
            # --- PATH B: SPATIAL BEAMFORMING (Benesty Universal) ---
            # H1 (Target), H2 (Jammer), H4 (Reverb), H5 (Interferer), H6 (Diffuse)
            else:
                S_hat[f] = self._path_b_spatial(f, y_t, R_nn[f], state)
                
        return S_hat

    def _path_a_spectral(self, f, y_t, R_n, state):
        """
        Fallback when spatial filtering is impossible.
        Uses Single-Channel Wiener Filter (Spectral Subtraction logic).
        """
        # 1. H0: Silence -> Hard Gate
        if state == 0:
            return 0.0 + 0j
        
        # 2. H7: Aliasing -> Pass through or Mild Attenuation
        if state == 7:
            return y_t[0] * 0.1 # Suppress high freq aliasing
            
        # 3. H3: Collinear (Target and Jammer are at 0 degrees)
        # Spatial filtering cannot separate them. We rely on magnitude.
        # Estimate Signal Power vs Noise Power
        
        # Power of input (use Ch 0)
        P_y = np.abs(y_t[0])**2
        
        # Power of Noise (Approximate from R_nn diagonal)
        P_n = np.real(R_n[0,0])
        
        # Wiener Gain G = SNR / (SNR + 1) = (Py - Pn) / Py
        if P_y > P_n:
            gain = (P_y - P_n) / (P_y + 1e-9)
        else:
            gain = 0.01 # Floor
            
        return y_t[0] * gain

    def _path_b_spatial(self, f, y_t, R_n, state):
        """
        Benesty's Universal Beamformer.
        w = inv(Phi + mu * R_xx) * d
        """
        # 1. Map State to Parameters (mu, Phi)
        
        # H2 (Tgt+Jam) or H5 (Jammer Only): Aggressive Nulling
        if state == 2 or state == 5:
            mu = 0.01 # "MU_NULL": Trust the nulls
            # Use Measured Noise Covariance (contains the Jammer)
            Phi = R_n 
            
        # H4 (Tgt+Diff) or H6 (Diffuse): Balanced Reduction
        elif state == 4 or state == 6:
            mu = 1.0 # "MU_WIENER": Balanced tradeoff
            # Use Theoretical Diffuse Model (Stable)
            Phi = self.Phi_diffuse[f] * np.real(R_n[0,0]) # Scale by noise power
            
        # H1 (Clean Target): Transparent
        else: # state == 1
            mu = 10.0 # "MU_CLEAN": Focus on distortionless response
            Phi = np.eye(2) * 1e-6 # Regularization only
            
        # 2. Construct Target Covariance Model (R_xx)
        # We model target as Rank-1: R_xx = sigma_s^2 * d * d^H
        # We need sigma_s^2. We use TSNR estimation.
        
        # Steering Vector [2x1]
        d = np.array([1.0, self.d_vec[f]], dtype=complex)
        
        # 3. Compute Beamformer Weights (Benesty Formulation)
        # W_opt = numerator / denominator
        # Matrix: J = Phi + mu * R_xx. 
        # Since R_xx is Rank 1, we can use Woodbury Identity or simplified MVDR form.
        
        # Simplified: Standard MVDR form using the composite Noise+Interference Matrix (Phi)
        # w = (Phi^-1 * d) / (d^H * Phi^-1 * d)
        
        # Regularize Phi for inversion
        Phi_reg = Phi + np.eye(2) * self.cfg.diagonal_load
        
        try:
            # Solve Phi_reg * x = d  => x = Phi_inv * d
            x_vec = np.linalg.solve(Phi_reg, d)
            
            # Denominator = d^H * x
            denom = np.vdot(d, x_vec).real + 1e-12
            
            w = x_vec / denom
            
        except np.linalg.LinAlgError:
            # Fallback if matrix singular
            w = np.array([0.5, 0.5]) 

        # 4. Apply Weights
        # Output = w^H * y
        s_hat = np.vdot(w, y_t)
        
        # 5. Post-Gating for Pure Interference (H5) or Diffuse (H6)
        # The beamformer tries to preserve the look direction. 
        # If H5 (Jammer), the energy coming from 0 deg is low, but we might leak side lobes.
        # Apply a spectral suppression gain.
        if state == 5 or state == 6:
            s_hat *= 0.1 # 20dB suppression for non-target frames
            
        return s_hat