import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

def dptf_sff_process_channel(audio, fs):
    """
    Applies DPTF-SFF enhancement with Spectral Weighting.
    """
    # --- Tunable Parameters ---
    K = 512 # Increased frequency resolution (Paper suggests high res)
    
    # 1. TUNING: Pole Radii (The "Sharpness" Knobs)
    # Closer to 1.0 = Sharper filters = More noise removal but more "robotic" tone
    # Previous values (0.99) were too wide for 10dB noise.
    r1 = 0.999
    r2 = 0.992
    
    # 2. TUNING: Noise Subtraction Factor (Over-subtraction)
    # How much to suppress the estimated noise?
    alpha_sub = 2.0 
    
    # --- Pre-processing ---
    # Pre-emphasis
    x = signal.lfilter([1, -0.97], [1], audio)
    N = len(x)
    n = np.arange(N)
    
    # --- Filter Setup ---
    # DPTF Coefficients
    a_coeffs = [1, r1 + r2, r1 * r2]
    b_coeffs = [1]
    
    # Storage for spectral bands
    band_outputs = np.zeros((K, N), dtype=np.complex64)
    envelopes = np.zeros((K, N))
    
    print(f"Analyzing {K} bands (r={r1}, {r2})...")
    
    # --- Analysis Loop ---
    for k in range(K):
        # Shift freq to pi (Nyquist)
        normalized_freq = 2 * np.pi * k / K
        modulator = np.exp(1j * normalized_freq * n)
        x_shifted = x * modulator
        
        # Filter
        y_filtered = signal.lfilter(b_coeffs, a_coeffs, x_shifted)
        
        # Demodulate
        y_band = y_filtered * np.conj(modulator)
        
        band_outputs[k, :] = y_band
        envelopes[k, :] = np.abs(y_band)

    # --- Synthesis with Spectral Weighting ---
    # To remove noise, we can't just sum. We must weigh the bands.
    # We estimate noise from the envelopes (assuming first 100ms is noise)
    noise_est_frames = int(0.1 * fs)
    noise_profile = np.mean(envelopes[:, :noise_est_frames], axis=1, keepdims=True)
    
    reconstructed_sum = np.zeros(N, dtype=np.complex64)
    
    # Processing bands
    for k in range(K):
        env = envelopes[k, :]
        noise_mu = noise_profile[k]
        
        # Calculate Gain based on Spectral Subtraction logic
        # SNR_k(n) approx = (Env^2 - Noise^2) / Env^2
        # Simple gain: G = max(0, 1 - alpha * Noise / Env)
        
        # Avoid divide by zero
        env_safe = np.maximum(env, 1e-10)
        
        # Gain calculation
        gain = 1.0 - (alpha_sub * noise_mu / env_safe)
        gain = np.maximum(gain, 0.01) # Floor at -40dB
        
        # Apply Gain to the complex band output
        reconstructed_sum += band_outputs[k, :] * gain

    # --- Post-processing ---
    enhanced_x = np.real(reconstructed_sum)
    enhanced_x = enhanced_x / K # Normalize
    
    # De-emphasis
    enhanced_audio = signal.lfilter([1], [1, -0.97], enhanced_x)
    
    return enhanced_audio

def dptf_sff_main(input_wav, output_wav):
    print(f"Loading {input_wav}...")
    fs, audio = wav.read(input_wav)
    
    # Normalize input
    original_dtype = audio.dtype
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
        
    if len(audio.shape) == 1:
        enhanced_audio = dptf_sff_process_channel(audio, fs)
    else:
        samples, channels = audio.shape
        enhanced_channels = []
        for ch in range(channels):
            print(f"--- Channel {ch+1}/{channels} ---")
            enhanced_ch = dptf_sff_process_channel(audio[:, ch], fs)
            enhanced_channels.append(enhanced_ch)
        enhanced_audio = np.column_stack(enhanced_channels)

    # Save
    max_val = np.max(np.abs(enhanced_audio))
    if max_val > 0:
        enhanced_audio = enhanced_audio / max_val * 0.9
        
    if np.issubdtype(original_dtype, np.integer):
        enhanced_audio = (enhanced_audio * 32767).astype(np.int16)
        
    wav.write(output_wav, fs, enhanced_audio)
    print(f"Saved to {output_wav}")

if __name__ == "__main__":
    dptf_sff_main("sample/enhanced_physics_hybrid.wav", "sample/enh_dptf.wav")