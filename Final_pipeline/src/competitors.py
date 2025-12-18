import os
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from src import config

# Constants matching your simulation settings
FFT_SIZE = 512
HOP = FFT_SIZE // 2

def compute_azimuth(source_pos, mic_center):
    """
    Calculates the azimuth angle of a source relative to the mic array center.
    """
    # vectors in x-y plane
    vec = np.array(source_pos[:2]) - np.array(mic_center[:2])
    # arctan2(y, x) returns radians
    return np.arctan2(vec[1], vec[0])

def run_competitors(run_name, output_dir=None):
    """
    Runs MVDR and LCMV on the simulated mixture.
    """
    if output_dir is None:
        output_dir = os.path.join(config.SIM_DIR, run_name)

    # 1. Load the Mixture created by simulation.py
    mix_path = os.path.join(output_dir, "mixture.wav")
    audio, fs = sf.read(mix_path)
    
    # Transpose to (n_mics, n_samples) for PyRoomAcoustics
    if audio.shape[0] > audio.shape[1]:
        audio = audio.T

    # 2. Reconstruct Geometry from Config
    # We use the exact same mic locations as the simulation
    mic_locs = np.array(config.MIC_LOCS_SIM).T 
    center = np.mean(mic_locs, axis=1)
    
    # Create a processing room (Anechoic for beamforming weights calculation)
    # We use a dummy room just to hold the array object
    room_proc = pra.ShoeBox(config.ROOM_DIM, fs=fs, max_order=0)
    room_proc.add_microphone_array(mic_locs)

    # 3. Calculate Steering Angles
    # Target Position from simulation.py
    pos_target = [2.45, 3.45, 1.5]
    angle_target = compute_azimuth(pos_target, center)

    # Interferer Position (The fixed one from simulation.py)
    pos_interf = [3.22, 3.06, 1.5]
    angle_interf = compute_azimuth(pos_interf, center)

    # --- COMPETITOR 1: MVDR ---
    # Goal: Preserve Target, Minimize total variance (noise)
    try:
        beamformer_mvdr = pra.Beamformer(
            room_proc.mic_array.R, 
            fs, 
            N=FFT_SIZE, 
            Lg=HOP
        )
        beamformer_mvdr.steer(
            theta=np.array([angle_target]), 
            profile=np.array([1.])
        )
        # Compute weights based on the actual audio (Signal+Noise covariance)
        beamformer_mvdr.raking_freq(audio) 
        out_mvdr = beamformer_mvdr.process(audio)
        
        # Save
        sf.write(os.path.join(output_dir, "est_mvdr.wav"), out_mvdr, fs)
    except Exception as e:
        print(f"[Competitors] MVDR Failed: {e}")

    # --- COMPETITOR 2: LCMV ---
    # Goal: Preserve Target (1), Null Interferer (0)
    try:
        beamformer_lcmv = pra.Beamformer(
            room_proc.mic_array.R, 
            fs, 
            N=FFT_SIZE, 
            Lg=HOP
        )
        # We enforce two constraints here
        beamformer_lcmv.steer(
            theta=np.array([angle_target, angle_interf]), 
            profile=np.array([1., 0.]) # 1 for target, 0 for interferer
        )
        beamformer_lcmv.raking_freq(audio)
        out_lcmv = beamformer_lcmv.process(audio)
        
        # Save
        sf.write(os.path.join(output_dir, "est_lcmv.wav"), out_lcmv, fs)
    except Exception as e:
        print(f"[Competitors] LCMV Failed: {e}")

    return True