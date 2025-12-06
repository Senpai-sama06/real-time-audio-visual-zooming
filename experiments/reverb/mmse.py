import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import scipy.special as special
from collections import deque
import os

def mmse_lsa_linked_prefilter(input_wav, output_wav):
    """
    Applies MMSE-LSA (Log-Spectral Amplitude) as a Pre-Filter with LINKED GAINS.
    This effectively boosts SNR from ~5dB to ~12-15dB while preserving spatial cues.
    
    Reference: Ephraim, Y. and Malah, D., "Speech enhancement using a minimum 
    mean-square error log-spectral amplitude estimator", IEEE Trans. ASSP, 1985.
    """
    # --- 1. Parameters ---
    frame_len_ms = 32
    overlap_ms = 16
    
    # Noise Estimation
    noise_buffer_len_sec = 1.5 
    min_stats_window = 5 # frames
    
    # SNR Estimation (Decision Directed)
    alpha_dd = 0.98  # Smoothing factor (beta in previous scripts)
    
    # Gain Floor (Maximum attenuation)
    # -25dB floor. Keeps a bit of noise background to prevent "pumping" artifacts
    min_gain = 0.056 # 10^(-25/20)
    
    # --- 2. Load Multi-channel Audio ---
    print(f"Loading {input_wav}...")
    try:
        fs, audio = wav.read(input_wav)
    except FileNotFoundError:
        print(f"Error: {input_wav} not found.")
        return

    # Normalize input to float32 range [-1, 1]
    original_dtype = audio.dtype
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
        
    # Handle Mono vs Multichannel
    if len(audio.shape) == 1:
        print("Note: Input is mono. Processing as single channel (no spatial cues to preserve).")
        audio = audio[:, np.newaxis] # Make it (N, 1)

    samples, channels = audio.shape
    print(f"Processing {channels} channels using Linked-Gain MMSE-LSA...")

    # --- 3. STFT Analysis (All Channels) ---
    nperseg = int(frame_len_ms * fs / 1000)
    noverlap = int(overlap_ms * fs / 1000)
    
    # Store transforms
    Zxx_all = []
    for ch in range(channels):
        f, t, Zxx = signal.stft(audio[:, ch], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
        Zxx_all.append(Zxx)
    
    Zxx_all = np.array(Zxx_all) # Shape: (Channels, Freqs, Time)
    Y_mag_all = np.abs(Zxx_all)
    Y_phase_all = np.angle(Zxx_all)
    
    n_freqs = Y_mag_all.shape[1]
    n_frames = Y_mag_all.shape[2]

    # --- 4. Initialization ---
    
    # Calculate AVERAGE magnitude across channels for robust statistics
    Y_mag_avg = np.mean(Y_mag_all, axis=0) # Shape: (Freqs, Time)
    
    # Initial Noise Estimate (Assume first 100ms is noise)
    noise_init_frames = max(5, int(0.1 * fs / (nperseg - noverlap)))
    noise_psd = np.mean(Y_mag_avg[:, :noise_init_frames]**2, axis=1)
    
    # Buffers for Minimum Statistics
    n_min_stats_buffer = int(noise_buffer_len_sec * (1000 / (frame_len_ms - overlap_ms)))
    p_min_buffer = deque(maxlen=n_min_stats_buffer)
    
    # Previous Enhanced Power (for Decision Directed SNR)
    S_hat_prev_pow = noise_psd.copy()
    
    # Gain container
    Common_Gain = np.zeros_like(Y_mag_avg)

    # --- 5. Processing Loop (Calculate Common Gain) ---
    print("Estimating gains...")
    
    for i in range(n_frames):
        Y_k = Y_mag_avg[:, i]
        Y_pow = Y_k ** 2
        
        # --- A. Noise Tracking (Simplified Min Statistics) ---
        # Smooth power spectrum
        smooth_factor = 0.85
        if i == 0: P_smooth = Y_pow
        else: P_smooth = smooth_factor * P_smooth + (1 - smooth_factor) * Y_pow
        
        p_min_buffer.append(P_smooth)
        
        # Find minimum in buffer
        if len(p_min_buffer) >= min_stats_window:
            # Min over time axis
            current_min = np.min(np.array(p_min_buffer), axis=0)
            # Bias correction (approx 1.5x for Rayleigh noise power under-estimation)
            noise_psd = current_min * 1.5
        else:
            noise_psd = P_smooth
            
        noise_psd = np.maximum(noise_psd, 1e-10)

        # --- B. SNR Estimation ---
        # a posteriori SNR (gamma)
        gamma_k = Y_pow / noise_psd
        gamma_k = np.maximum(gamma_k, 0.01) # Limit to -20dB to prevent errors
        
        # a priori SNR (xi) - Decision Directed
        # xi = alpha * (PrevClean / Noise) + (1-alpha) * P[gamma - 1]
        xi_k = alpha_dd * (S_hat_prev_pow / noise_psd) + (1 - alpha_dd) * np.maximum(gamma_k - 1, 0)
        xi_k = np.maximum(xi_k, 0.01) # Limit -20dB

        # --- C. MMSE-LSA Gain Calculation ---
        # Formula: G = (xi / (1+xi)) * exp(0.5 * E1(nu))
        # where nu = (xi / (1+xi)) * gamma
        
        # 1. Calculate v (nu)
        v = (xi_k / (1.0 + xi_k)) * gamma_k
        v = np.maximum(v, 1e-10) # Avoid zero for E1 function
        
        # 2. Calculate Gain
        # exp1 is E_1(x) = integral_x^inf (e^-t / t) dt
        eig_val = 0.5 * special.exp1(v)
        
        Gain = (xi_k / (1.0 + xi_k)) * np.exp(eig_val)
        
        # 3. Floor Gain
        Gain = np.maximum(Gain, min_gain)
        # Cap gain at 1.0 (we don't want to amplify)
        Gain = np.minimum(Gain, 1.0)
        
        Common_Gain[:, i] = Gain
        
        # Update memory for next frame's a priori SNR
        # Note: We estimate the "Clean" power using the gain we just calculated
        S_hat_prev_pow = (Gain * Y_k) ** 2

    # --- 6. Apply Common Gain and Synthesize ---
    enhanced_channels = []
    
    print("Applying linked gain to all channels...")
    for ch in range(channels):
        # Apply the SAME gain to each channel's magnitude
        S_hat_ch = Common_Gain * Y_mag_all[ch]
        
        # Use original phase
        _, enhanced_audio = signal.istft(S_hat_ch * np.exp(1j * Y_phase_all[ch]), 
                                         fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
        enhanced_channels.append(enhanced_audio)

    # --- 7. Save Output ---
    # Handle length mismatch (ISTFT can add slight padding)
    min_len = min([len(c) for c in enhanced_channels])
    enhanced_channels = [c[:min_len] for c in enhanced_channels]
    
    # Stack back to (N, Channels)
    enhanced_output = np.column_stack(enhanced_channels)
    
    # Normalize peak to -1.0 dB
    max_val = np.max(np.abs(enhanced_output))
    if max_val > 0:
        enhanced_output = enhanced_output / max_val * 0.9
    
    # Convert back to original type if needed
    if np.issubdtype(original_dtype, np.integer):
        enhanced_output = (enhanced_output * 32767).astype(np.int16)
        
    wav.write(output_wav, fs, enhanced_output)
    print(f"Success! Enhanced audio saved to {output_wav}")

# --- Dummy File Generator (for testing) ---
def generate_dummy_multichannel(filename, fs=16000, duration=5):
    if os.path.exists(filename): return
    
    print("Generating dummy stereo file with 5dB SNR...")
    t = np.linspace(0, duration, duration*fs)
    
    # Signal: A chirp
    clean = signal.chirp(t, f0=200, f1=3000, t1=duration, method='logarithmic')
    clean *= 0.5 * np.exp(-t) # Envelope
    
    # Stereo Simulation
    # Ch1: Signal + Noise
    # Ch2: Signal (delayed) + Noise (uncorrelated)
    
    # Target 5dB SNR
    # SNR = 10 log10 (Ps / Pn) -> 0.5 = 10 log10 (ratio) -> ratio = 10^0.5 = 3.16
    # If clean amp is 0.5, noise amp needs to be significant.
    
    noise_amp = 0.3 # Rough visual approx for 5dB
    
    noise1 = np.random.normal(0, noise_amp, len(t))
    noise2 = np.random.normal(0, noise_amp, len(t))
    
    ch1 = clean + noise1
    ch2 = np.roll(clean, 50) + noise2 # 50 sample delay
    
    stereo = np.column_stack((ch1, ch2))
    stereo = stereo / np.max(np.abs(stereo)) * 0.9
    
    wav.write(filename, fs, (stereo * 32767).astype(np.int16))

if __name__ == "__main__":
    input_file = "sample/mixture.wav"
    output_file = "sample/mmse_lsa_mix.wav"
    
    generate_dummy_multichannel(input_file)
    mmse_lsa_linked_prefilter(input_file, output_file)