import numpy as np
import scipy.signal
import soundfile as sf
import os

# ------------------------
# Constants
# ------------------------
FS = 16000
N_FFT = 256
N_HOP = 128
MIC_DIST = 0.08
C_SPEED = 343.0
ANGLE_TARGET = 90.0
N_MICS = 2

# Thresholds for Switching Logic
THRESH_ANISOTROPY = 3.0   # If lambda1/lambda2 > 3, noise field is directional
THRESH_SPATIAL = 0.85     # Cosine similarity threshold to distinguish Target vs Interferer
SIGMA_LOAD = 1e-3         # Diagonal loading factor for RMVB

SAVE_DIR = "/home/rpzrm/global/projects/real-time-audio-visual-zooming/experiments/reverb/sample"

# ========================================================
# Steering Vector
# ========================================================
def get_steering_vector_single(f, angle_deg, d, c):
    theta = np.deg2rad(angle_deg)

    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f

    v = np.array([
        [np.exp(-1j * omega * tau1)],
        [np.exp(-1j * omega * tau2)]
    ], dtype=complex)

    # Normalize 0th element to phase 0 for stability
    return v / (v[0] + 1e-10)

# ========================================================
# Blind Switching Beamformer (SMVB)
# ========================================================
def blind_switching_bf(Y, f_bins):
    """
    Implements Blind SMVB: Switches between LCMV and RMVB based on 
    Eigen-Rank and Spatial Alignment, using only the Mixture covariance.
    """
    n_freq = Y.shape[1]
    n_frames = Y.shape[2]

    S_out = np.zeros((n_freq, n_frames), dtype=complex)
    
    # Desired response for LCMV: [Gain=1 for Target, Gain=0 for Interferer]
    desired_lcmv = np.array([[1], [0]], dtype=np.complex64)

    for i in range(n_freq):
        f_hz = f_bins[i]

        # 1. Low-freq bypass (Spatial aliasing/noise floor protection)
        if f_hz < 200:
            S_out[i, :] = Y[0, i, :]
            continue

        Y_f = Y[:, i, :]  # (2, T)
        
        # 2. Estimate Blind Covariance Matrix Ryy (from Mixture)
        # Ryy = E[y y^H]
        R_yy = (Y_f @ Y_f.conj().T) / n_frames

        # 3. Eigen Analysis
        eigvals, eigvecs = np.linalg.eigh(R_yy)
        
        # Sort eigenvalues descending (lambda1 >= lambda2)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        lambda1, lambda2 = eigvals[0], eigvals[1]
        v_dominant = eigvecs[:, 0].reshape(2, 1) # Dominant spatial direction

        # 4. Compute Anisotropy (Rank Proxy)
        eta = lambda1 / (lambda2 + 1e-10)

        # 5. Get Target Steering Vector
        v_tgt = get_steering_vector_single(f_hz, ANGLE_TARGET, MIC_DIST, C_SPEED)

        # 6. Spatial Check: Is the dominant source the Target?
        # Compute cosine similarity between Target Vector and Dominant Eigenvector
        # |v_tgt^H . v_dom|
        spatial_sim = np.abs(v_tgt.conj().T @ v_dominant)

        # ====================================================
        # SWITCHING LOGIC
        # ====================================================
        
        # CONDITION: 
        # If Anisotropy is HIGH (Directional Source) 
        # AND Spatial Similarity is LOW (Dominant source is NOT the target)
        # THEN -> It is a strong Interferer. Use LCMV (Hard Null).
        if eta > THRESH_ANISOTROPY and spatial_sim < THRESH_SPATIAL:
            # --- MODE: LCMV (Hard Nulling) ---
            v_int = v_dominant # Assume dominant vector is the interferer
            
            # Form Constraint Matrix C = [v_tgt, v_int]
            C = np.column_stack((v_tgt, v_int))
            
            # Check condition number to avoid numerical instability
            if np.linalg.cond(C) > 100:
                # Fallback if matrix is singular (sources too close)
                w = v_tgt / N_MICS
            else:
                try:
                    # w = C * inv(C^H * C) * f (Simplified inverse solution)
                    # Solves C^H * w = desired
                    w = np.linalg.solve(C.conj().T, desired_lcmv)
                except:
                    w = v_tgt / N_MICS
                    
        else:
            # --- MODE: RMVB (Robust Loading) ---
            # Used when:
            # 1. Noise is diffuse (Low Eta)
            # 2. Target is the dominant source (High Eta but High Spatial Sim)
            
            # Add Diagonal Loading
            R_loaded = R_yy + SIGMA_LOAD * np.trace(R_yy) * np.eye(N_MICS)
            
            try:
                # w = (R + sigma*I)^-1 * v_tgt
                # Normalized by denominator (standard MVDR formulation)
                R_inv_d = np.linalg.solve(R_loaded, v_tgt)
                w = R_inv_d / (v_tgt.conj().T @ R_inv_d + 1e-10)
            except:
                w = v_tgt / N_MICS

        # Apply Beamformer weights
        S_out[i, :] = (w.conj().T @ Y_f).squeeze()

    return S_out


# ========================================================
# MAIN: Blind SMVB
# ========================================================
def main():
    print("=== BLIND SMVB (Statistical Switching) ===")
    print(f"Thresholds -> Anisotropy: {THRESH_ANISOTROPY}, Spatial Sim: {THRESH_SPATIAL}")

    mix_path = f"{SAVE_DIR}/mixture.wav"

    if not os.path.exists(mix_path):
        print(f"Missing mixture file at {mix_path}")
        return

    # Load mixture
    y_mix, sr = sf.read(mix_path, dtype='float32')
    if sr != FS:
        print(f"Warning: SR mismatch. Expected {FS}, got {sr}")
    
    Y = y_mix.T  # (2, S)

    # STFT
    f_bins, t_bins, Y_mix = scipy.signal.stft(
        Y, fs=FS, nperseg=N_FFT, noverlap=N_HOP
    )

    # -----------------------------
    # RUN BLIND SMVB
    # -----------------------------
    # No masks are passed here. The function derives stats from Y_mix.
    S_out = blind_switching_bf(Y_mix, f_bins)

    # ISTFT
    _, s_out = scipy.signal.istft(
        S_out, fs=FS, nperseg=N_FFT, noverlap=N_HOP
    )
    
    # Normalize
    s_out /= np.max(np.abs(s_out) + 1e-10)

    out_path = f"{SAVE_DIR}/output_blind_smvb.wav"
    sf.write(out_path, s_out, FS)
    print(f"Saved: {out_path}")

            
if __name__ == "__main__":
    main()