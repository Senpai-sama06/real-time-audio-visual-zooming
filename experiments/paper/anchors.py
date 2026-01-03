import numpy as np

class PhysicsAnchors:
    """
    Computes and stores the theoretical anchor points in the Complex Coherence Plane.
    Anchor 1: Target (Unit Circle)
    Anchor 2: Diffuse (Real Axis / Sinc)
    """
    def __init__(self, config):
        self.n_bins = config.n_fft // 2 + 1
        self.freqs = np.linspace(0, config.fs/2, self.n_bins)
        
        # Angular frequency
        omega = 2 * np.pi * self.freqs
        
        # --- 1. Target Anchor (Unit Circle) ---
        # Phase shift = omega * tau
        tau = (config.mic_dist * np.cos(config.target_angle)) / config.c
        self.target = np.exp(1j * omega * tau)
        
        # --- 2. Diffuse Anchor (Sinc Function) ---
        # Gamma_diffuse = sinc(omega * d / c)
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = omega * config.mic_dist / config.c
            # numpy sinc is sin(pi*x)/(pi*x), we need sin(x)/x
            self.diffuse = np.sin(arg) / (arg + 1e-9)
            
            # Correct DC (0 Hz) -> limit is 1.0
            self.diffuse[0] = 1.0
            
        # --- 3. Aliasing Limit ---
        # f_alias = c / (2 * d)
        self.alias_freq = config.c / (2 * config.mic_dist)
        self.alias_bin_idx = np.searchsorted(self.freqs, self.alias_freq)

    def is_aliased(self, bin_idx):
        return bin_idx >= self.alias_bin_idx