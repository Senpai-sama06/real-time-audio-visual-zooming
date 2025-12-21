'''
python world.py --no-reverb --dataset ljspeech --n 1
'''

import numpy as np
import soundfile as sf
import os
import glob
import random
import argparse
import librosa
import kagglehub
import pyroomacoustics as pra
from scipy.signal import fftconvolve

# --- 1. Constants ---
FS = 16000          
ROOM_DIM = [4.9, 4.9, 4.9] 
RT60_TARGET = 0.5   
SNR_TARGET_DB = 5
SIR_TARGET_DB = 0   

# Mic Array (Center of room, 8cm spacing)
MIC_LOCS = np.array([
    [2.41, 2.45, 1.5], # Mic 1 (Left)
    [2.49, 2.45, 1.5]  # Mic 2 (Right)
]).T 

# Fixed Source Positions
POS_TARGET = [2.45, 3.45, 1.5]  # ~90 degrees
POS_INTERF_FIXED = [3.22, 3.06, 1.5] # ~40 degrees

# --- 2. Helper Functions ---

def get_audio_files(dataset_name, n_needed, min_duration=5.0):
    """Fetches n_needed random audio files that are at least min_duration seconds long."""
    print(f"--- Fetching {n_needed} files (>= {min_duration}s) from: {dataset_name} ---")
    files = []
    try:
        if dataset_name == 'librispeech':
            path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
            files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
        elif dataset_name == 'musan':
            path = kagglehub.dataset_download("dogrose/musan-dataset")
            files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
        else: 
            # Default: ljspeech
            path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
            wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
            files = glob.glob(os.path.join(wav_path, "*.wav"))
        
        if len(files) == 0: 
            raise ValueError(f"No files found for {dataset_name}")

        # Shuffle to ensure random selection
        random.shuffle(files)
        
        valid_files = []
        print("Scanning files for duration requirements...")
        
        # Iterate through shuffled files and pick those that meet the duration requirement
        for f in files:
            if len(valid_files) >= n_needed:
                break
            try:
                # Use sf.info to check duration without loading the full file (much faster)
                info = sf.info(f)
                if info.duration >= min_duration:
                    valid_files.append(f)
            except Exception as e:
                continue

        # Handle case where we didn't find enough valid files
        if len(valid_files) < n_needed:
            print(f"Warning: Only found {len(valid_files)} valid files >= {min_duration}s. Duplicating to reach {n_needed}.")
            if len(valid_files) == 0:
                 raise ValueError("No files found meeting the duration requirement.")
            while len(valid_files) < n_needed:
                valid_files += valid_files
            valid_files = valid_files[:n_needed]
        
        return valid_files

    except Exception as e:
        print(f"Error getting data: {e}")
        return []

def add_awgn(signal, snr_db):
    """
    Adds noise to a signal at a specific SNR.
    Returns the noisy signal.
    """
    sig_power = np.mean(signal ** 2)
    if sig_power == 0: return signal
    noise_power = sig_power / (10 ** (snr_db / 10))
    # noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noise = np.zeros(signal.shape)
    return signal + noise, noise

def main():
    # --- 0. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Cocktail Party Simulator (Task 5)")
    
    # 1. Reverb Flag (Default: True)
    parser.add_argument('--reverb', action='store_true', default=True, help="Enable reverb (default)")
    parser.add_argument('--no-reverb', action='store_false', dest='reverb', help="Disable reverb (Anechoic)")
    
    # 2. Number of Interferers (Default: 1)
    parser.add_argument('--n', type=int, default=1, help="Number of interferers (Default 1: 1 fixed at 40deg)")
        
    # 3. Dataset (Default: musan)
    parser.add_argument('--dataset', type=str, default='librispeech', choices=['musan', 'librispeech', 'ljspeech'])
    
    args = parser.parse_args()
    
    print(f"Configuration: Dataset={args.dataset} | Reverb={args.reverb} | Interferers={args.n}")

    # --- 1. Load Audio ---
    # We need 1 Target + args.n Interferers
    total_sources = 1 + args.n
    files = get_audio_files(args.dataset, total_sources)
    if len(files) < total_sources: return

    # --- NEW CODE START: Print Selected Files ---
    print("\n" + "="*40)
    print("SELECTED AUDIO SOURCES:")
    print(f"  [TARGET]      : {os.path.basename(files[0])}")
    for i, f_path in enumerate(files[1:]):
        print(f"  [INTERFERER {i+1}]: {os.path.basename(f_path)}")
    print("="*40 + "\n")
    # --- NEW CODE END ---

    # Load and normalize length
    sigs = []
    min_len = float('inf')
    
    for f in files:
        y, _ = librosa.load(f, sr=FS, mono=True)
        sigs.append(y)
        if len(y) < min_len: min_len = len(y)
    
    # Truncate all to minimum length
    sigs = [s[:min_len] for s in sigs]
    
    target_sig = sigs[0]
    interferer_sigs = sigs[1:] # List of n interferer signals

    # --- 2. Setup Room ---
    if args.reverb:
        # Reverb Mode: Calculate absorption from RT60 (Allen & Berkley Image Method)
        e_absorption, max_order = pra.inverse_sabine(RT60_TARGET, ROOM_DIM)
        materials = pra.Material(e_absorption)
        # max_order usually defaults high (e.g. 15-20) for reverb
        m_order = 15 
    else:
        # Anechoic Mode: Full absorption (1.0) and 0 reflections
        materials = pra.Material(1.0)
        m_order = 0

    room = pra.ShoeBox(ROOM_DIM, fs=FS, materials=materials, max_order=m_order)
    room.add_microphone_array(MIC_LOCS)
    
    # --- 3. Add Sources ---
    # Source 0: Target (Fixed 90 deg)
    room.add_source(POS_TARGET)
    
    # Source 1+: Interferers
    if args.n > 0:
        # First interferer is Fixed (40 deg)
        room.add_source(POS_INTERF_FIXED)
        
        # Remaining interferers (if n > 1) are Random
        for _ in range(args.n - 1):
            rx = random.uniform(1.0, ROOM_DIM[0]-1.0)
            ry = random.uniform(1.0, ROOM_DIM[1]-1.0)
            room.add_source([rx, ry, 1.5])

    # --- 4. Compute RIRs ---
    print("Computing RIRs...")
    room.compute_rir()
    
    # --- 5. Convolution & Mixing ---
    # Helper to pad/trim
    def get_convolved(sig, rir):
        return fftconvolve(sig, rir, mode='full')[:min_len]

    # Process Target
    target_ch1 = get_convolved(target_sig, room.rir[0][0])
    target_ch2 = get_convolved(target_sig, room.rir[1][0])

    # Process Interferers (Summation)
    interf_ch1_total = np.zeros(min_len)
    interf_ch2_total = np.zeros(min_len)

    if args.n > 0:
        for i, i_sig in enumerate(interferer_sigs):
            # Source index in room is i + 1 (because 0 is target)
            src_idx = i + 1
            i_ch1 = get_convolved(i_sig, room.rir[0][src_idx])
            i_ch2 = get_convolved(i_sig, room.rir[1][src_idx])
            
            interf_ch1_total += i_ch1
            interf_ch2_total += i_ch2

    # --- 6. SIR Control ---
    # Calculate energy at Mic 1
    p_target = np.mean(target_ch1 ** 2)
    p_interf = np.mean(interf_ch1_total ** 2)
    
    gain = 1
    if args.n > 0 and p_interf > 0:
        # SIR = 10 log10 (P_t / (g^2 * P_i))
        # g = sqrt( P_t / (P_i * 10^(SIR/10)) )
        # gain = np.sqrt(p_target / (p_interf * (10**(SIR_TARGET_DB/10))))
        print(f"Applying Gain {gain:.4f} to Interferers for {SIR_TARGET_DB}dB SIR")
        
        interf_ch1_total *= gain
        interf_ch2_total *= gain
    
    # Clean Mixture (No Noise)
    clean_mix_ch1 = target_ch1 + interf_ch1_total
    clean_mix_ch2 = target_ch2 + interf_ch2_total

    # --- 7. Add Noise (SNR Control) ---
    print(f"Adding AWGN at {SNR_TARGET_DB}dB SNR")
    final_ch1, noise_ch1 = add_awgn(clean_mix_ch1, SNR_TARGET_DB)
    final_ch2, noise_ch2 = add_awgn(clean_mix_ch2, SNR_TARGET_DB)

    # --- 8. Normalization & Saving (Task 5) ---
    
    # Stack Stereo Signals

    # noise
    stereo_noise = np.stack([noise_ch1, noise_ch2], axis=1)

    # Mixture (Noisy)
    stereo_mix = np.stack([final_ch1, final_ch2], axis=1)
    
    # Target Reference (Clean, Reverberant)
    stereo_target = np.stack([target_ch1, target_ch2], axis=1)
    
    # Interference Reference (Clean, Reverberant)
    stereo_interf = np.stack([interf_ch1_total, interf_ch2_total], axis=1)
    
    # Global Peak Normalization (Based on the Noisy Mixture)
    # We apply the SAME scalar to all files to preserve the energy ratios (SIR/SNR).
    peak = np.max(np.abs(stereo_mix)) + 1e-9
    
    stereo_mix /= peak
    stereo_target /= peak
    stereo_interf /= peak
    stereo_noise /= peak

    # Save Files
    sf.write("sample/mixture.wav", stereo_mix, FS)
    sf.write("sample/target.wav", stereo_target, FS)
    sf.write("sample/interference.wav", stereo_interf, FS)
    sf.write("sample/noise.wav", stereo_noise, FS)
    
    print(f"Simulation Complete.")
    print(f"Generated Files:")
    print(f"  1. mixture.wav      (Target + Interf + Reverb + Noise)")
    print(f"  2. target.wav       (Target + Reverb [Reference])")
    print(f"  3. interference.wav (Interf + Reverb [Reference])")
    print(f"  4. noise.wav (noise)")

if __name__ == "__main__":
    main()