import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import stft
from scipy.io import wavfile
import sys
import os

# Import our custom system
from classifier import SpatioSpectralClassifier
from config import SystemConfig # Imported to verify sampling rate

def load_mixture(filename):
    """
    Loads a wav file and checks constraints.
    Returns: fs, audio [Samples, 2]
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
        
    fs, audio = wavfile.read(filename)
    
    # 1. Convert to Float [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
        
    # 2. Check Channel Count
    if audio.ndim != 2 or audio.shape[1] != 2:
        print(f"Error: Input file must be Stereo (2 channels). Got shape {audio.shape}")
        sys.exit(1)
        
    return fs, audio

if __name__ == "__main__":
    # --- CONFIGURATION ---
    WAV_FILE = "mixture.wav"  # <--- Put your file name here
    N_FFT = 512
    HOP_LEN = 128
    
    # --- 1. LOAD AUDIO ---
    print(f"Loading {WAV_FILE}...")
    fs, audio = load_mixture(WAV_FILE)
    
    # Check if file sampling rate matches our SystemConfig default
    # (Important: Physics constants in config.py depend on this)
    default_cfg = SystemConfig()
    if fs != default_cfg.fs:
        print(f"WARNING: File FS ({fs}Hz) != System FS ({default_cfg.fs}Hz).")
        print("Please update 'fs' in config.py to match your audio file,")
        print("or the spatial physics (aliasing/anchors) will be wrong.")
    
    # --- 2. COMPUTE STFT ---
    print("Computing STFT...")
    # Channel 1
    f, t, Zxx = stft(audio[:,0], fs=fs, nperseg=N_FFT, noverlap=HOP_LEN)
    # Channel 2
    _, _, Zyy = stft(audio[:,1], fs=fs, nperseg=N_FFT, noverlap=HOP_LEN)
    
    # Stack for processing: [Bins, Frames, 2]
    Y_stft = np.stack([Zxx, Zyy], axis=-1) 
    
    # --- 3. RUN CLASSIFIER ---
    print("Running Spatio-Spectral Hypothesis Switch...")
    classifier = SpatioSpectralClassifier(target_angle_rad=1.57)
    
    n_frames = Y_stft.shape[1]
    results = np.zeros((Y_stft.shape[0], n_frames), dtype=int)
    
    # Process Frame-by-Frame
    for i in range(n_frames):
        results[:, i] = classifier.process_frame(Y_stft[:, i, :])

    # --- 4. VISUALIZATION (3 PANELS) ---
    print("Plotting Results...")
    fig, ax = plt.subplots(1, 3, figsize=(12, 12), sharex=True)
    
    # Panel 1: Spectrogram Channel 1
    ax[0].pcolormesh(t, f, 10*np.log10(np.abs(Zxx)**2 + 1e-12), shading='auto', cmap='inferno')
    ax[0].set_title(f"Channel 1 Spectrogram (Left)")
    ax[0].set_ylabel("Frequency [Hz]")
    
    # Panel 2: Spectrogram Channel 2
    ax[1].pcolormesh(t, f, 10*np.log10(np.abs(Zyy)**2 + 1e-12), shading='auto', cmap='inferno')
    ax[1].set_title(f"Channel 2 Spectrogram (Right)")
    ax[1].set_ylabel("Frequency [Hz]")
    
    # Panel 3: Hypothesis Map
    # Color Definition
    cmap_colors = [
        'black',  # 0: H0 Silence
        'lime',   # 1: H1 Target
        'red',    # 2: H2 Tgt+Jammer
        'yellow', # 3: H3 Collinear (Reserved)
        'cyan',   # 4: H4 Tgt+Diffuse
        'orange', # 5: H5 Jammer Only
        'magenta',# 6: H6 Diffuse Only
        'white'   # 7: H7 Aliasing
    ]
    labels = ["H0: Silence", "H1: Target", "H2: Tgt+Jam", "H3: Collinear", 
              "H4: Tgt+Diff", "H5: Jammer", "H6: Diffuse", "H7: Aliasing"]
              
    cmap = mcolors.ListedColormap(cmap_colors)
    bounds = np.arange(9) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax[2].pcolormesh(t, f, results, cmap=cmap, norm=norm, shading='nearest')
    ax[2].set_title("Spatio-Spectral Hypothesis Map")
    ax[2].set_ylabel("Frequency [Hz]")
    ax[2].set_xlabel("Time [s]")
    
    # Legend
    cbar = plt.colorbar(im, ax=ax[2], ticks=np.arange(8))
    cbar.ax.set_yticklabels(labels)
    
    plt.tight_layout()
    plt.show()
    print("Done.")