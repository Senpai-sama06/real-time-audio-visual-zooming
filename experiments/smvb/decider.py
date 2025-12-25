import numpy as np

class BeamformerSwitch:
    def __init__(self, num_bins, alpha=0.7, thr_coh=0.6, thr_pe=0.5):
        """
        Stereo Switching Block for GSC Beamformer.
        
        Args:
            num_bins (int): Number of frequency bins.
            alpha (float): Smoothing factor (0.7 recommended).
            thr_coh (float): Threshold for Coherence (Diffuseness).
            thr_pe (float): Threshold for Phase Error (Interference).
        """
        self.alpha = alpha
        self.thr_coh = thr_coh
        self.thr_pe = thr_pe
        
        # Internal State Memory (Previous PSDs)
        self.P11 = np.zeros(num_bins, dtype=float)
        self.P22 = np.zeros(num_bins, dtype=float)
        self.P12 = np.zeros(num_bins, dtype=complex)
        
        # Constants
        self.MODE_RMVB = 0
        self.MODE_LCMV = 1

    def process_frame(self, X_stereo):
        """
        Input: Stereo STFT frame.
               Expected Shape: (2, num_bins) 
               - Row 0: Mic 1 (Reference)
               - Row 1: Mic 2
        
        Returns:
            switch_mode: Array of flags (0=RMVB, 1=LCMV)
            null_sign: Array of signs (+1=Left, -1=Right, 0=None)
        """
        
        # --- 1. Error Handling (The Fallback) ---
        # If input is not (2, bins), we cannot do stereo logic.
        # Fallback: Force RMVB (Safe Mode) for all bins.
        if X_stereo.ndim != 2 or X_stereo.shape[0] != 2:
            # Create safe defaults matching internal bin count
            safe_mode = np.zeros(self.P11.shape, dtype=int) + self.MODE_RMVB
            safe_sign = np.zeros(self.P11.shape, dtype=int)
            return safe_mode, safe_sign

        # --- 2. Unpack Stereo Input ---
        X1 = X_stereo[0]  # Mic 1
        X2 = X_stereo[1]  # Mic 2
        
        # --- 3. Feature Calculation (Internal State Update) ---
        
        # Instantaneous Power
        inst_P11 = np.abs(X1)**2
        inst_P22 = np.abs(X2)**2
        inst_P12 = X1 * np.conj(X2)
        
        # Recursive Smoothing
        self.P11 = self.alpha * self.P11 + (1 - self.alpha) * inst_P11
        self.P22 = self.alpha * self.P22 + (1 - self.alpha) * inst_P22
        self.P12 = self.alpha * self.P12 + (1 - self.alpha) * inst_P12
        
        # Compute Features
        coherence_sq = (np.abs(self.P12)**2) / (self.P11 * self.P22 + 1e-12)
        phase_angle = np.angle(self.P12)
        pe_abs = np.abs(phase_angle)
        
        # --- 4. The Decision Logic ---
        
        # Diffuseness Check
        is_diffuse = coherence_sq < self.thr_coh
        is_directional = ~is_diffuse
        
        # Interferer Check (Directional + High Phase Error)
        is_interferer = is_directional & (pe_abs > self.thr_pe)
        
        # --- 5. Output Generation ---
        
        # Default to RMVB (Safe)
        switch_mode = np.zeros_like(X1, dtype=int) + self.MODE_RMVB
        
        # Trigger LCMV only for Interferers
        switch_mode[is_interferer] = self.MODE_LCMV
        
        # Calculate Null Steering Sign (+1 Left, -1 Right)
        null_sign = np.zeros_like(X1, dtype=int)
        mask_lcmv = (switch_mode == self.MODE_LCMV)
        
        # Only calc sign where we actually need to null
        null_sign[mask_lcmv] = np.sign(phase_angle[mask_lcmv])
        
        return switch_mode, null_sign
    




'''
---------------
Example usage
---------------


# Initialization
switcher = BeamformerSwitch(num_bins=257)

# Inside your loop
# X_current_frame shape is (2, 257)
modes, signs = switcher.process_frame(X_current_frame)

if modes[10] == 0:
    print("Bin 10: Safe Mode (RMVB)")
else:
    print(f"Bin 10: Attack Mode (LCMV), Null Direction: {signs[10]}")
'''