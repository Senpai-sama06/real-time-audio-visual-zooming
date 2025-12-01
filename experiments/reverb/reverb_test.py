import numpy as np
import soundfile as sf
import os
import glob
import random
import librosa
import kagglehub
import pyroomacoustics as pra

# --- 1. Constants & Specs ---
FS = 16000          
ROOM_DIM = [5, 4, 3] 
RT60_TARGET = 0.5   
SNR_TARGET_DB = 5   # Task 2: 5dB SNR

# Dual Mic Array (from world.py)
MIC_LOCS = np.array([
    [2.41, 2.45, 1.5], # Mic 1 (Left Channel)
    [2.49, 2.45, 1.5]  # Mic 2 (Right Channel)
]).T 

# --- KEY CHANGE: Source Position for 40 degrees ---
# This coordinate corresponds to an approximate 40 degree Angle of Arrival 
# relative to the mic array's broadside (0 degrees is along the y-axis).
SOURCE_POS = [3.22, 3.06, 1.5] # New Source Position

# --- 2. Helper Functions ---

def get_single_file(dataset_name='ljspeech'):
    """Fetches a single file."""
    print(f"--- Fetching Dataset: {dataset_name} ---")
    files = []
    try:
        # (Same dataset fetching logic as world.py)
        if dataset_name == 'librispeech':
            path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
            files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
        elif dataset_name == 'musan':
            path = kagglehub.dataset_download("dogrose/musan-dataset")
            files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
        else: 
            path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
            wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
            files = glob.glob(os.path.join(wav_path, "*.wav"))
        
        if not files: raise ValueError("No files found.")
        return random.choice(files)
    except Exception as e:
        print(f"Error getting data: {e}")
        return None

def add_awgn(signal_channel, snr_db, L):
    """
    Adds Additive White Gaussian Noise to a single channel.
    """
    sig_power = np.mean(signal_channel ** 2)
    if sig_power == 0: return signal_channel
    
    # Calculate required noise power for the target SNR: P_noise = P_signal / 10^(SNR/10)
    noise_power = sig_power / (10 ** (snr_db / 10))
    
    # Generate Noise
    noise = np.random.normal(0, np.sqrt(noise_power), L)
    return signal_channel + noise

# --- 3. Main Script ---

def main_aoa():
    # 1. Load Audio
    audio_path = get_single_file('ljspeech') 
    if audio_path is None: return

    clean_sig, _ = librosa.load(audio_path, sr=FS, mono=True)
    
    # 2. Setup and Simulate Room (Image Method)
    e_absorption, max_order = pra.inverse_sabine(RT60_TARGET, ROOM_DIM)
    
    room = pra.ShoeBox(ROOM_DIM, fs=FS, materials=pra.Material(e_absorption), max_order=15)
    room.add_source(SOURCE_POS, signal=clean_sig)
    room.add_microphone_array(MIC_LOCS) # Dual Mic Setup

    room.compute_rir()
    room.simulate()
    
    # Get the reverberant signal (shape: 2, L_out)
    reverb_signals = room.mic_array.signals
    L_out = reverb_signals.shape[1]

    # 3. Add 5dB AWGN
    print(f"\n--- Simulation: Reverb + {SNR_TARGET_DB}dB SNR, Source at 40 degrees ---")
    
    # Initialize array for noisy output
    noisy_reverb_signals = np.zeros_like(reverb_signals)
    
    # Apply noise to EACH channel individually
    for i in range(reverb_signals.shape[0]):
        noisy_reverb_signals[i, :] = add_awgn(reverb_signals[i, :], SNR_TARGET_DB, L_out)
    
    # 4. Normalization and Save
    
    # Combine channels into a stereo array (L_out, 2)
    stereo_mix = np.stack(noisy_reverb_signals, axis=1)
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(stereo_mix))
    if peak > 0:
        stereo_mix /= peak
    
    output_filename = "task3_dual_mic_40deg_noisy.wav"
    sf.write(os.path.join(os.getcwd(), output_filename), stereo_mix, FS)
    print(f"Simulation Complete. Final Output: {output_filename} (2-channel/Stereo, 40 deg AoA, 5dB AWGN)")

if __name__ == "__main__":
    main_aoa()