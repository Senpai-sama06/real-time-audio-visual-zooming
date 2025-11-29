'''
usage : 
python world.py --no-reverb --n 10
'''


import numpy as np
import soundfile as sf
import os
import kagglehub
import glob
import random
import datetime
import librosa
import argparse
import pyroomacoustics as pra
from pystoi import stoi
from pesq import pesq

# --- 1. Constants & Specs ---
FS = 16000          
ROOM_DIM = [4.9, 4.9, 4.9] 
RT60_TARGET = 0.5   
SIR_TARGET_DB = 0   
SNR_TARGET_DB = 5   

# Mic Array: Center of room, 8cm spacing
MIC_LOCS = np.array([
    [2.41, 2.45, 1.5], # Mic 1 (Left)
    [2.49, 2.45, 1.5]  # Mic 2 (Right)
]).T 

SOURCE_TARGET_POS = [2.45, 3.45, 1.5] # 90 degrees
SOURCE_INTERF_POS = [3.22, 3.06, 1.5] # 40 degrees

# --- 2. Helper Functions ---

def get_audio_files(dataset_name, n_needed):
    """Unified file fetcher for all datasets."""
    files = []
    print(f"--- Fetching Dataset: {dataset_name} ---")
    try:
        if dataset_name == 'librispeech':
            path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
            files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
        elif dataset_name == 'musan':
            path = kagglehub.dataset_download("dogrose/musan-dataset")
            files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
        else: 
            # Default: LJSpeech
            path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
            wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
            files = glob.glob(os.path.join(wav_path, "*.wav"))

        if len(files) < n_needed:
            if len(files) > 0:
                print(f"Warning: Only found {len(files)} files. Duplicating to reach {n_needed}.")
                while len(files) < n_needed:
                    files += files
            else:
                raise ValueError(f"No files found for {dataset_name}")
        
        return random.sample(files, n_needed)

    except Exception as e:
        print(f"Error getting data: {e}")
        return []

def load_audio_and_resample(file_path, target_fs, target_len=None, min_duration=4.0):
    """Loads audio, resamples, and pads/trims to exact target_len."""
    try:
        y, orig_fs = sf.read(file_path, dtype='float32')
        if len(y.shape) > 1: y = np.mean(y, axis=1) # Mono
        if orig_fs != target_fs:
            y = librosa.resample(y, orig_sr=orig_fs, target_sr=target_fs)
        
        if target_len is None:
            desired_samples = int(min_duration * target_fs)
            if len(y) < desired_samples:
                repeats = int(np.ceil(desired_samples / len(y)))
                y = np.tile(y, repeats)[:desired_samples]
        else:
            if len(y) < target_len:
                 repeats = int(np.ceil(target_len / len(y)))
                 y = np.tile(y, repeats)[:target_len]
            else:
                y = y[:target_len]
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(int(min_duration*target_fs))

def add_awgn(signal, snr_db):
    sig_power = np.mean(signal ** 2)    
    if sig_power == 0: return signal
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def calculate_metrics(clean, degraded, fs):
    min_len = min(len(clean), len(degraded))
    clean = clean[:min_len]
    degraded = degraded[:min_len]
    try: d_stoi = stoi(clean, degraded, fs, extended=False)
    except: d_stoi = 0.0
    try: d_pesq = pesq(fs, clean, degraded, 'wb')
    except: d_pesq = 0.0
    return d_stoi, d_pesq

# --- 3. Main Script ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverb', action='store_true')
    parser.add_argument('--no-reverb', action='store_false', dest='reverb')
    parser.add_argument('--n', type=int, default=1, help='Extra interferers')
    parser.add_argument('--dataset', type=str, default='ljspeech', choices=['ljspeech', 'librispeech', 'musan'])
    args = parser.parse_args()

    # --- 1. Setup Room ---
    print(f"--- Settings: {args.dataset} | {'Reverb' if args.reverb else 'Anechoic'} | +{args.n} Extra Src ---")
    
    if args.reverb:
        e_absorption = pra.inverse_sabine(RT60_TARGET, ROOM_DIM)
        
        # FIX FOR TypeError: float() argument must be a string or a real number, not 'tuple'
        if isinstance(e_absorption, (tuple, np.ndarray)):
            # Assume we only want the first coefficient for a flat absorber material
            e_absorption = e_absorption[0] 
            
        materials = pra.Material(float(e_absorption))
        max_order = 10 
    else:
        materials = pra.Material(1.0)
        max_order = 0

    room = pra.ShoeBox(ROOM_DIM, fs=FS, materials=materials, max_order=max_order)
    room.add_microphone_array(MIC_LOCS)

    # --- 2. Load Audio ---
    total_sources_needed = 2 + args.n 
    files = get_audio_files(args.dataset, total_sources_needed)
    if not files or len(files) < total_sources_needed:
        print("CRITICAL ERROR: Not enough audio files.")
        return

    # Load Target (Master Length)
    target_sig = load_audio_and_resample(files[0], FS, target_len=None) 
    L = len(target_sig)

    # Load Interferer 1 (Fixed)
    interf1_sig = load_audio_and_resample(files[1], FS, target_len=L)
    
    # Load Extra Interferers (Random)
    extra_interferers = []
    for i in range(args.n):
        sig = load_audio_and_resample(files[2+i], FS, target_len=L)
        extra_interferers.append(sig)

    # --- 3. Add Sources to Room ---
    # Index 0: Target
    room.add_source(SOURCE_TARGET_POS, signal=target_sig)
    
    # Index 1: Fixed Interferer
    room.add_source(SOURCE_INTERF_POS, signal=interf1_sig)
    
    # Index 2+: Extra Interferers
    for i, sig in enumerate(extra_interferers):
        rx = random.uniform(1.0, ROOM_DIM[0]-1.0)
        ry = random.uniform(1.0, ROOM_DIM[1]-1.0)
        room.add_source([rx, ry, 1.5], signal=sig)

    # --- 4. Compute RIRs ---
    print("Simulating RIRs...")
    room.compute_rir()

    # --- 5. Mix with SIR Control ---
    mix_ch1 = np.zeros(L)
    mix_ch2 = np.zeros(L)
    target_ref = np.zeros(L)
    interf_ref = np.zeros(L)
    
    gain_scalar = 1.0 
    
    # --- CALCULATE GAIN (Using Mic 1) ---
    t_rir_1 = room.rir[0][0]
    t_comp_1 = np.convolve(target_sig, t_rir_1)[:L]
    
    i_total_raw_1 = np.zeros(L)
    
    # Fixed Interferer
    i1_rir_1 = room.rir[0][1]
    i_total_raw_1 += np.convolve(interf1_sig, i1_rir_1)[:L]
    
    # Extra Interferers
    for k, sig in enumerate(extra_interferers):
        ik_rir_1 = room.rir[0][2+k]
        i_total_raw_1 += np.convolve(sig, ik_rir_1)[:L]
        
    # Gain for SIR = 0dB
    p_t = np.mean(t_comp_1**2)
    p_i = np.mean(i_total_raw_1**2)
    
    if p_i > 0:
        gain_scalar = np.sqrt(p_t / p_i)
    else:
        gain_scalar = 0.0
        
    print(f"Calculated Gain for SIR 0dB: {gain_scalar:.4f}")

    # --- CONSTRUCT MIXTURES ---
    
    for m_idx in range(2): # 0=Mic1, 1=Mic2
        # Target
        t_comp = np.convolve(target_sig, room.rir[m_idx][0])[:L]
        
        # Total Interference
        i_total = np.zeros(L)
        i_total += np.convolve(interf1_sig, room.rir[m_idx][1])[:L]
        for k, sig in enumerate(extra_interferers):
            i_total += np.convolve(sig, room.rir[m_idx][2+k])[:L]
            
        # Apply Gain
        i_final = i_total * gain_scalar
        
        # Mix + Noise
        clean_mix = t_comp + i_final
        noisy_mix = add_awgn(clean_mix, SNR_TARGET_DB)
        
        if m_idx == 0:
            mix_ch1 = noisy_mix
            target_ref = t_comp # Mic 1 Target Reference
            interf_ref = i_final # Mic 1 Interference Reference (Scaled)
        else:
            mix_ch2 = noisy_mix

    # --- 6. Normalization & Saving ---
    peak = max(np.max(np.abs(mix_ch1)), np.max(np.abs(mix_ch2))) + 1e-9
    
    mix_ch1 /= peak
    mix_ch2 /= peak
    
    target_ref_norm = target_ref / (np.max(np.abs(target_ref)) + 1e-9)
    interf_ref_norm = interf_ref / (np.max(np.abs(interf_ref)) + 1e-9)
    
    # Combine into Stereo (2-channel)
    stereo_mix = np.stack([mix_ch1, mix_ch2], axis=1)

    # Metrics
    val_stoi, val_pesq = calculate_metrics(target_ref_norm, mix_ch1, FS)
    
    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "reverb" if args.reverb else "anechoic"
    output_dir = os.path.join(f"simulation_results", f"{args.dataset}_{mode_str}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # WRITE STEREO MIXTURE
    sf.write(os.path.join(output_dir, "mixture.wav"), stereo_mix, FS)
    
    # WRITE MONO REFERENCES
    sf.write(os.path.join(output_dir, "target_reference.wav"), target_ref_norm, FS)
    sf.write(os.path.join(output_dir, "interference_reference.wav"), interf_ref_norm, FS)
    
    with open(os.path.join(output_dir, "info.txt"), "w") as f:
        f.write(f"Dataset: {args.dataset}\nMode: {mode_str}\n")
        f.write(f"STOI: {val_stoi}\nPESQ: {val_pesq}\n")
        f.write(f"Channels: mixture.wav (Stereo), target_ref (Mono), interf_ref (Mono)")

    print(f"DONE. Output folder: {output_dir}")
    print("Files created: mixture.wav (Stereo), target_reference.wav, interference_reference.wav")

if __name__ == "__main__":
    main()