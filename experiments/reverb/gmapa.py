import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from collections import deque

def gmapa_process_channel(audio_channel, fs):
    """
    Applies GMAPA to a single channel of audio data.
    Returns the enhanced time-domain audio for that channel.
    """
    # --- 1. Parameters ---
    frame_len_ms = 32
    overlap_ms = 16
    
    # STW Parameters (Optimized from Paper)
    alpha_max = 2.195
    b = -0.7787
    c = 21.18
    
    Mc = 30 # Temporal Grouping window
    noise_buffer_len_sec = 1.5 
    beta = 0.98 # Decision-Directed smoothing
    min_gain = 0.1 # -20 dB floor
    
    # --- 2. STFT Analysis ---
    nperseg = int(frame_len_ms * fs / 1000)
    noverlap = int(overlap_ms * fs / 1000)
    nfft = nperseg
    
    f, t, Zxx = signal.stft(audio_channel, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    Y_mag = np.abs(Zxx)
    Y_phase = np.angle(Zxx)
    rows, cols = Y_mag.shape
    
    # --- 3. Initialization ---
    S_hat_mag = np.zeros_like(Y_mag)
    
    # Initial Noise Estimate (Assume first 5 frames are noise)
    noise_psd = np.mean(Y_mag[:, :5]**2, axis=1)
    # Ensure noise_psd has correct shape (rows,)
    if noise_psd.ndim > 1: noise_psd = noise_psd.flatten()
        
    n_noise_frames = int(noise_buffer_len_sec * (1000 / (frame_len_ms - overlap_ms)))
    p_min_buffer = deque(maxlen=n_noise_frames)
    gamma_buffer = deque(maxlen=Mc)
    
    S_hat_prev_pow = noise_psd.copy()
    
    # --- 4. Frame Processing ---
    for i in range(cols):
        Y_k = Y_mag[:, i]
        Y_pow = Y_k ** 2
        
        # --- Noise Estimation (Simplified Min Stats) ---
        smooth_factor = 0.85
        if i == 0:
            P_smooth = Y_pow
        else:
            P_smooth = smooth_factor * P_smooth + (1 - smooth_factor) * Y_pow
            
        p_min_buffer.append(P_smooth)
        
        if len(p_min_buffer) >= 5:
            current_min = np.min(np.array(p_min_buffer), axis=0)
            noise_psd = current_min * 1.5 
        else:
            noise_psd = P_smooth
            
        noise_psd = np.maximum(noise_psd, 1e-10)

        # --- SNR Statistics ---
        gamma_k = Y_pow / noise_psd
        gamma_k = np.maximum(gamma_k, 0.01)
        
        xi_k = beta * (S_hat_prev_pow / noise_psd) + (1 - beta) * np.maximum(gamma_k - 1, 0)
        xi_k = np.maximum(xi_k, 0.01)

        # --- Adaptive Weight (Alpha) ---
        frame_avg_gamma = np.mean(gamma_k)
        gamma_buffer.append(frame_avg_gamma)
        gamma_bar = np.mean(gamma_buffer)
        
        exponent = -b * (gamma_bar - c)
        exponent = np.clip(exponent, -50, 50)
        alpha = alpha_max / (1 + np.exp(exponent))
        
        # --- GMAPA Gain Calculation ---
        term1 = xi_k
        A = 2 * alpha - 1
        B = alpha + xi_k
        C = xi_k / gamma_k
        
        sqrt_arg = xi_k**2 + A * B * C
        sqrt_arg = np.maximum(sqrt_arg, 0) # Safety clamp
        
        term2 = np.sqrt(sqrt_arg)
        denominator = 2 * (alpha + xi_k)
        
        Gain = (term1 + term2) / denominator
        Gain = np.maximum(Gain, min_gain)
        
        # Reconstruct
        S_hat_mag[:, i] = Gain * Y_k
        S_hat_prev_pow = S_hat_mag[:, i]**2

    # --- 5. Synthesis ---
    _, enhanced_channel = signal.istft(S_hat_mag * np.exp(1j * Y_phase), fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    
    return enhanced_channel

def gmapa_enhance_multichannel(input_wav, output_wav):
    print(f"Loading {input_wav}...")
    fs, audio = wav.read(input_wav)
    
    # Normalize input
    original_dtype = audio.dtype
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
        
    # Check dimensions
    # Scipy reads mono as (N,) and stereo as (N, 2)
    if len(audio.shape) == 1:
        # Mono case
        print("Detected Mono audio.")
        enhanced_audio = gmapa_process_channel(audio, fs)
    else:
        # Multi-channel case
        samples, channels = audio.shape
        print(f"Detected {channels}-channel audio.")
        
        enhanced_channels = []
        for ch in range(channels):
            print(f"Processing Channel {ch+1}/{channels}...")
            channel_data = audio[:, ch]
            enhanced_ch = gmapa_process_channel(channel_data, fs)
            enhanced_channels.append(enhanced_ch)
        
        # Stack channels back together (N, channels)
        # Handle potential length mismatch from ISTFT (padding)
        min_len = min([len(c) for c in enhanced_channels])
        enhanced_channels = [c[:min_len] for c in enhanced_channels]
        
        enhanced_audio = np.column_stack(enhanced_channels)

    # Scale and Save
    print("Reconstructing and saving...")
    # Normalize peak to avoid clipping
    max_val = np.max(np.abs(enhanced_audio))
    if max_val > 0:
        enhanced_audio = enhanced_audio / max_val * 0.9
        
    if np.issubdtype(original_dtype, np.integer):
        enhanced_audio = (enhanced_audio * 32767).astype(np.int16)
    
    wav.write(output_wav, fs, enhanced_audio)
    print(f"Done! Saved to {output_wav}")

# --- Execution ---
if __name__ == "__main__":
    input_file = "sample/mixture.wav" 
    output_file = "sample/premix.wav"
    
    # Generate dummy stereo file if input doesn't exist
    import os
    if not os.path.exists(input_file):
        print("Generating dummy stereo file...")
        fs = 16000
        t = np.linspace(0, 5, 5*fs)
        clean = np.sin(2*np.pi*440*t) * np.exp(-3*t)
        # Channel 1: Clean + Noise
        ch1 = clean + np.random.normal(0, 0.4, clean.shape)
        # Channel 2: Clean (delayed) + Different Noise
        ch2 = np.roll(clean, 100) + np.random.normal(0, 0.4, clean.shape)
        
        stereo = np.column_stack((ch1, ch2))
        stereo = stereo / np.max(np.abs(stereo)) * 0.8
        wav.write(input_file, fs, (stereo * 32767).astype(np.int16))

    gmapa_enhance_multichannel(input_file, output_file)