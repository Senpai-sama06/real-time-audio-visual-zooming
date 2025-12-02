import numpy as np
import soundfile as sf
import os
import glob
import random
import librosa
import kagglehub
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from src import config

def get_audio_files(dataset_name, n_needed):
    """
    Fetches n_needed random audio files from Kaggle datasets.
    """
    print(f"--- Fetching {n_needed} files from: {dataset_name} ---")
    files = []
    try:
        if dataset_name == 'librispeech':
            # Downloads LibriSpeech
            path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
            files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
        elif dataset_name == 'musan':
            # Downloads MUSAN
            path = kagglehub.dataset_download("dogrose/musan-dataset")
            files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
        else: 
            # Default: ljspeech
            path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
            wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
            files = glob.glob(os.path.join(wav_path, "*.wav"))
        
        # Handle case where we need more files than exist (duplicate if needed)
        if len(files) == 0: 
            raise ValueError(f"No files found for {dataset_name}")
            
        if len(files) < n_needed:
            print(f"Warning: Only {len(files)} files found. Duplicating to reach {n_needed}.")
            while len(files) < n_needed:
                files += files
        
        return random.sample(files, n_needed)
    except Exception as e:
        print(f"Error getting data: {e}")
        return []

def add_awgn(signal, snr_db):
    """
    Adds Gaussian white noise to a signal at a specific SNR.
    """
    sig_power = np.mean(signal ** 2)
    if sig_power == 0: return signal
    
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal +noise

def generate_scene(run_name, dataset='ljspeech', reverb=True, n_interferers=1, snr_target=5):
    """
    Generates a simulated audio scene.
    
    Args:
        run_name (str): Identifier for the run (e.g. 'test1').
        dataset (str): Dataset source name.
        reverb (bool): Whether to apply reverberation.
        n_interferers (int): Number of interfering speakers.
        snr_target (int): Target Signal-to-Noise Ratio (Gaussian noise) in dB.
        
    Returns:
        str: Path to the generated mixture file.
    """
    # 1. Setup Output Directory
    # Saves to: data/simulated/{run_name}/
    save_dir = os.path.join(config.SIM_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[SIM] Generating '{run_name}' in {save_dir}...")
    print(f"[SIM] Config: Dataset={dataset} | Reverb={reverb} | Interferers={n_interferers} | SNR={snr_target}dB")

    # 2. Get Files & Load Audio
    # We need 1 Target + n Interferers
    total_sources = 1 + n_interferers
    files = get_audio_files(dataset, total_sources)
    
    if len(files) < total_sources:
        print("[SIM] Error: Not enough audio files retrieved.")
        return None

    # Load audio and determine minimum common length
    sigs = []
    min_len = float('inf')
    
    for f in files:
        # Load at 16kHz Mono
        y, _ = librosa.load(f, sr=config.FS, mono=True)
        sigs.append(y)
        if len(y) < min_len: min_len = len(y)
    
    # Truncate all signals to the minimum length so they mix perfectly
    sigs = [s[:min_len] for s in sigs]
    
    target_sig = sigs[0]
    interferer_sigs = sigs[1:] # List could be empty if n=0

    # 3. Setup Room (Physics)
    if reverb:
        # Reverb Mode: Calculate absorption from RT60
        e_absorption, max_order = pra.inverse_sabine(config.RT60_TARGET, config.ROOM_DIM)
        materials = pra.Material(e_absorption)
        m_order = 15 
    else:
        # Anechoic Mode: Full absorption (1.0) -> No reflections
        materials = pra.Material(1.0)
        m_order = 0

    room = pra.ShoeBox(config.ROOM_DIM, fs=config.FS, materials=materials, max_order=m_order)
    
    # Add Microphone Array (Transposed for PyRoomAcoustics format)
    # config.MIC_LOCS_SIM is [[x1, y1, z1], [x2, y2, z2]], PRA expects [[x1, x2], [y1, y2], [z1, z2]]
    room.add_microphone_array(np.array(config.MIC_LOCS_SIM).T)
    
    # Add Target Source (Fixed Position ~90 degrees)
    pos_target = [2.45, 3.45, 1.5]
    room.add_source(pos_target)
    
    # Add Interferers
    if n_interferers > 0:
        # First interferer is Fixed at ~40 degrees
        pos_interf_fixed = [3.22, 3.06, 1.5]
        room.add_source(pos_interf_fixed)
        
        # Remaining interferers are Random
        for _ in range(n_interferers - 1):
            rx = random.uniform(1.0, config.ROOM_DIM[0]-1.0)
            ry = random.uniform(1.0, config.ROOM_DIM[1]-1.0)
            room.add_source([rx, ry, 1.5])

    # 4. Compute RIRs
    print("[SIM] Computing Room Impulse Responses (Ray Tracing)...")
    room.compute_rir()
    
    # Helper to convolve and trim
    def get_convolved(sig, rir):
        return fftconvolve(sig, rir, mode='full')[:min_len]

    # --- 5. Convolution & Mixing ---
    
    # A. Process Target
    # room.rir[mic_idx][source_idx]
    target_ch1 = get_convolved(target_sig, room.rir[0][0])
    target_ch2 = get_convolved(target_sig, room.rir[1][0])

    # B. Process Interferers (Summation)
    interf_ch1_total = np.zeros(min_len)
    interf_ch2_total = np.zeros(min_len)

    if n_interferers > 0:
        for i, i_sig in enumerate(interferer_sigs):
            # Source index in room is i + 1 (because 0 is target)
            src_idx = i + 1
            i_ch1 = get_convolved(i_sig, room.rir[0][src_idx])
            i_ch2 = get_convolved(i_sig, room.rir[1][src_idx])
            
            interf_ch1_total += i_ch1
            interf_ch2_total += i_ch2

    # --- 6. SIR Control (Signal-to-Interference Ratio) ---
    p_target = np.mean(target_ch1 ** 2)
    p_interf = np.mean(interf_ch1_total ** 2)
    
    if n_interferers > 0 and p_interf > 0:
        # Calculate gain required to achieve SIR_TARGET_DB
        # SIR = 10 * log10( P_target / (Gain^2 * P_interf) )
        desired_ratio = 10**(config.SIR_TARGET_DB/10)
        gain = np.sqrt(p_target / (p_interf * desired_ratio))
        
        print(f"[SIM] Applying Gain {gain:.4f} to Interferers for {config.SIR_TARGET_DB}dB SIR")
        interf_ch1_total *= gain
        interf_ch2_total *= gain
    
    # Create Clean Mixture (Target + Interference, No Noise yet)
    clean_mix_ch1 = target_ch1 + interf_ch1_total
    clean_mix_ch2 = target_ch2 + interf_ch2_total

    # --- 7. SNR Control (Signal-to-Noise Ratio) ---
    print(f"[SIM] Adding AWGN at {snr_target}dB SNR")
    final_ch1 = add_awgn(clean_mix_ch1, snr_target)
    final_ch2 = add_awgn(clean_mix_ch2, snr_target)

    # --- 8. Normalization & Saving ---
    
    # Stack Stereo Signals
    stereo_mix = np.stack([final_ch1, final_ch2], axis=1)
    stereo_target = np.stack([target_ch1, target_ch2], axis=1)
    stereo_interf = np.stack([interf_ch1_total, interf_ch2_total], axis=1)
    
    # Global Peak Normalization based on the noisy mixture
    peak = np.max(np.abs(stereo_mix)) + 1e-9
    
    stereo_mix /= peak
    stereo_target /= peak
    stereo_interf /= peak

    # Save Files
    mix_path = os.path.join(save_dir, "mixture.wav")
    tgt_path = os.path.join(save_dir, "target.wav")
    int_path = os.path.join(save_dir, "interference.wav")
    
    sf.write(mix_path, stereo_mix, config.FS)
    sf.write(tgt_path, stereo_target, config.FS)
    sf.write(int_path, stereo_interf, config.FS)
    
    print(f"[SIM] Simulation Complete.")
    print(f"[SIM] Files Saved to: {save_dir}")
    print(f"      - mixture.wav")
    print(f"      - target.wav")
    print(f"      - interference.wav")

    return mix_path