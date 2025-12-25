import numpy as np

def calculate_lcmv_weights(v_tgt, v_neural, steering_sign=0):
    """
    Calculates weights for a Hard Null (LCMV) Beamformer using a pre-computed
    interference vector (from Neural Net) and a safety check (from Analytic Switch).

    Parameters:
    -----------
    v_tgt : np.ndarray (M,)
        Target steering vector (Analytic, e.g., Broadside).
    v_neural : np.ndarray (M,)
        Dominant eigenvector from the Neural Covariance Matrix.
        (Represents the interference direction).
    steering_sign : int
        -1 (Right), 1 (Left), or 0 (Unknown).
        From the Analytic Decision Block. Used for safety checking.

    Returns:
    --------
    w : np.ndarray (M,)
        Complex weights. Returns None if safety check fails.
    """
    M = len(v_tgt)
    
    # --- 1. Safety Check: Does Neural Vector match Analytic Physics? ---
    if steering_sign != 0:
        # Calculate Phase Difference between mics for the Neural Vector
        # (Assuming linear array, phase diff maps to angle)
        phase_diff = np.angle(v_neural[1] * np.conj(v_neural[0]))
        
        # Check alignment:
        # If steering_sign is +1 (Left), phase_diff should be Positive (Standard lag convention)
        # If steering_sign is -1 (Right), phase_diff should be Negative
        # (Note: Verify your specific mic cabling sign convention!)
        
        if np.sign(phase_diff) != np.sign(steering_sign):
            # DANGER: Neural Net says Left, Physics says Right.
            # This is likely a "Ghost" reflection. Do NOT null.
            return None

    # --- 2. The Orthogonal Projection (Your Logic) ---
    # We want w such that: w.H @ v_tgt = 1  AND  w.H @ v_neural = 0
    
    v_int = v_neural # Alias for readability
    
    # Project Target onto the subspace orthogonal to Interference
    # u = v_tgt - proj_coeff * v_int
    dot_vn_vt = np.vdot(v_int, v_tgt)
    dot_vn_vn = np.vdot(v_int, v_int).real
    
    # Singularity Check: Is Interference parallel to Target?
    if dot_vn_vn < 1e-10:
        return None 
        
    proj_coeff = dot_vn_vt / dot_vn_vn
    u = v_tgt - (proj_coeff * v_int)

    # --- 3. White Noise Gain / Singularity Check ---
    # If u is tiny, it means Target and Interference are nearly identical.
    # Nulling the interference would kill the target.
    if np.linalg.norm(u) < 1e-3:
        return None

    # --- 4. Normalization for Distortionless Response ---
    # w.H @ v_tgt = 1
    current_gain = np.vdot(u, v_tgt)
    
    if np.abs(current_gain) < 1e-6:
        return None
        
    w = u / np.conj(current_gain)
    
    return w