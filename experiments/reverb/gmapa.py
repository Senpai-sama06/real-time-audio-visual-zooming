import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from collections import deque

def gmapa_enhance(input_wav, output_wav):
    """
    Implements the Generalized Maximum a Posteriori Spectral Amplitude (GMAPA)
    algorithm for speech enhancement.
    
    Reference: Tsao, Y., & Lai, Y. H. (2015). Generalized maximum a posteriori 
    spectral amplitude estimation for speech enhancement. Speech Communication.
    """
    
    # --- 1. Parameters (from Paper Section 4.2 & Standard Values) ---
    
    # STFT parameters
    frame_len_ms = 32      # Frame length in ms
    overlap_ms = 16        # Overlap in ms
    
    # STW (SNR-to-Weight) Function Parameters (Optimized values from Paper)
    alpha_max = 2.195      # Maximum weight for Prior
    b = -0.7787            # Slope of sigmoid
    c = 21.18              # Center point of sigmoid (dB)
    
    # Temporal Grouping (TG) Parameter
    Mc = 30                # Sliding window size for alpha calculation (frames)
    
    # Noise Estimation Parameters (Minimum Statistics simplified)
    noise_buffer_len_sec = 1.5 
    
    # Decision-Directed Parameters
    beta = 0.98            # Smoothing factor for a priori SNR
    
    # Gain flooring (to prevent absolute silence/artifacts)
    min_gain = 0.1         # -20 dB floor
    
    # --- 2. Load Audio ---
    print(f"Loading {input_wav}...")
    fs, audio = wav.read(input_wav)
    
    # Normalize audio to -1.0 to 1.0 range
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # --- 3. STFT Analysis ---
    nperseg = int(frame_len_ms * fs / 1000)
    noverlap = int(overlap_ms * fs / 1000)
    nfft = nperseg
    
    f, t, Zxx = signal.stft(audio, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    # Y_mag: Noisy Magnitude Spectrum, Y_phase: Noisy Phase
    Y_mag = np.abs(Zxx)
    Y_phase = np.angle(Zxx)
    
    rows, cols = Y_mag.shape
    
    # --- 4. Initialization ---
    
    # Output Magnitude container
    S_hat_mag = np.zeros_like(Y_mag)
    
    # Initial Noise Estimate (Assume first 5 frames are noise)
    noise_psd = np.mean(Y_mag[:, :5]**2, axis=1)
    
    # Buffers for Minimum Statistics Noise Estimation
    n_noise_frames = int(noise_buffer_len_sec * (1000 / (frame_len_ms - overlap_ms)))
    p_min_buffer = deque(maxlen=n_noise_frames)
    
    # Buffer for Temporal Grouping (Average Gamma calculation)
    gamma_buffer = deque(maxlen=Mc)
    
    # Previous Enhanced Spectrum for Decision-Directed approach
    S_hat_prev_pow = noise_psd # Initialize with noise estimate
    
    print("Processing frames...")
    
    # --- 5. Frame-by-Frame Processing ---
    
    for i in range(cols):
        # Current Noisy Amplitude and Power
        Y_k = Y_mag[:, i]
        Y_pow = Y_k ** 2
        
        # --- A. Noise Estimation (Simplified Minimum Statistics) ---
        # Smooth power spec for noise tracking
        smooth_factor = 0.85
        if i == 0:
            P_smooth = Y_pow
        else:
            P_smooth = smooth_factor * P_smooth + (1 - smooth_factor) * Y_pow
            
        p_min_buffer.append(P_smooth)
        
        # Estimate noise PSD as the minimum of the smoothed power over the buffer
        # (Multiplied by a bias factor, typically 1.5 to 2.0 for MS over-estimation compensation)
        if len(p_min_buffer) >= 5:
            current_min = np.min(np.array(p_min_buffer), axis=0)
            noise_psd = current_min * 1.5 
        else:
            # Fallback for first few frames
            noise_psd = P_smooth
            
        # Avoid division by zero
        noise_psd = np.maximum(noise_psd, 1e-10)

        # --- B. Calculate SNR Statistics ---
        
        # 1. A Posteriori SNR (Gamma)
        gamma_k = Y_pow / noise_psd
        gamma_k = np.maximum(gamma_k, 0.01) # Clamp to avoid near-zero/neg
        
        # 2. A Priori SNR (Xi) - Decision Directed Approach
        # xi = beta * (PrevEnhanced / Noise) + (1 - beta) * max(gamma - 1, 0)
        xi_k = beta * (S_hat_prev_pow / noise_psd) + (1 - beta) * np.maximum(gamma_k - 1, 0)
        xi_k = np.maximum(xi_k, 0.01) # Lower bound -20dB

        # --- C. Calculate Adaptive Weight Alpha (The GMAPA Core) ---
        
        # Temporal Grouping: Average a posteriori SNR over last Mc frames
        # We average gamma across frequencies first to get frame-level SNR, then average over time
        # Note: The paper defines gamma_bar as mean of gamma over frames
        frame_avg_gamma = np.mean(gamma_k) 
        gamma_buffer.append(frame_avg_gamma)
        
        gamma_bar = np.mean(gamma_buffer) # This is the "mean a posteriori SNR" for the window
        
        # Convert to dB for the sigmoid function? 
        # The paper's Table 2 shows gamma_bar values like 17.73, 22.76. 
        # These look like dB values (10log10) or high linear values. 
        # Given standard SNR ranges, 17-22 linear is ~12-13dB. 
        # Let's assume linear based on standard literature, but check paper context.
        # Paper Table 2: Input SNR 0dB -> Gamma 17.73. This is definitely LINEAR scale.
        # (If it were dB, average gamma for 0dB input would be closer to 0-5 dB).
        
        # STW Function (Eq 23)
        # exponent = -b * (gamma_bar - c)
        exponent = -b * (gamma_bar - c)
        
        # Clamp exponent to prevent overflow
        exponent = np.clip(exponent, -50, 50)
        
        alpha = alpha_max / (1 + np.exp(exponent))
        
        # --- D. Calculate GMAPA Gain (Eq 22) ---
        
        # G = (xi + sqrt(xi^2 + (2*alpha - 1)*(alpha + xi)*(xi/gamma))) / (2*(alpha + xi))
        
        term1 = xi_k
        
        # Term inside square root
        # (2*alpha - 1)
        A = 2 * alpha - 1
        # (alpha + xi)
        B = alpha + xi_k
        # (xi / gamma)
        C = xi_k / gamma_k
        
        sqrt_arg = xi_k**2 + A * B * C
        
        # Safety check: if alpha is small (<0.5) and gamma is small, sqrt_arg might be negative.
        # This corresponds to MLSA constraint. We clamp sqrt_arg to 0.
        sqrt_arg = np.maximum(sqrt_arg, 0)
        
        term2 = np.sqrt(sqrt_arg)
        
        denominator = 2 * (alpha + xi_k)
        
        Gain = (term1 + term2) / denominator
        
        # Apply Gain Floor
        Gain = np.maximum(Gain, min_gain)
        
        # --- E. Reconstruct Frame ---
        S_hat_mag[:, i] = Gain * Y_k
        
        # Save power for next frame's xi calculation
        S_hat_prev_pow = S_hat_mag[:, i]**2

    # --- 6. Synthesis (ISTFT) ---
    print("Reconstructing audio...")
    _, enhanced_audio = signal.istft(S_hat_mag * np.exp(1j * Y_phase), fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    # Scale back to PCM range
    enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.9 # Peak normalize
    
    if np.issubdtype(np.int16, np.integer):
        enhanced_audio = (enhanced_audio * 32767).astype(np.int16)
    
    wav.write(output_wav, fs, enhanced_audio.astype(np.int16))
    print(f"Enhanced audio saved to {output_wav}")

# --- Execution Block ---
if __name__ == "__main__":
    # Replace these filenames with your actual files
    input_file = "sample/mixture.wav" 
    output_file = "sample/output_gmapa_mixture.wav"
    
    # Create a dummy file for testing if it doesn't exist
    import os
    if not os.path.exists(input_file):
        print(f"Note: '{input_file}' not found. Generating a dummy 5dB SNR file for demonstration.")
        fs = 16000
        t = np.linspace(0, 5, 5*fs)
        clean = np.sin(2*np.pi*440*t) * np.exp(-3*t) # Decaying sine
        noise = np.random.normal(0, 0.4, clean.shape) # Noise
        mixed = clean + noise
        # Normalize
        mixed = mixed / np.max(np.abs(mixed)) * 0.8
        wav.write(input_file, fs, (mixed * 32767).astype(np.int16))

    gmapa_enhance(input_file, output_file)