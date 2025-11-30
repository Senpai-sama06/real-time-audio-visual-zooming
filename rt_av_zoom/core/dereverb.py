import numpy as np
import soundfile as sf
import os
import argparse
import glob
from nara_wpe.wpe import wpe
from nara_wpe.utils import istft, stft

# --- Constants ---
FS = 16000
N_FFT = 512
N_HOP = 256

# --- Dynamic Path Finding ---
BASE_RESULTS_DIR = os.path.join(os.getcwd(), "simulation_results")

def get_latest_run_dir():
    if not os.path.exists(BASE_RESULTS_DIR):
        return None
    search_pattern = os.path.join(BASE_RESULTS_DIR, '*_*_*') 
    all_runs = sorted(glob.glob(search_pattern), reverse=True)
    return all_runs[0] if all_runs else None

DEFAULT_OUTDIR = get_latest_run_dir()

def apply_wpe(y_mix, taps=10, delay=3, iterations=3):
    """
    Robust WPE application.
    Expected Input: (Channels, Samples)
    """
    print(f"  [WPE] Input Time-Domain Shape: {y_mix.shape}")
    
    # 1. Compute STFT
    stft_opts = dict(size=N_FFT, shift=N_HOP)
    try:
        Y = stft(y_mix, **stft_opts)
    except Exception as e:
        print(f"  [Error] STFT failed: {e}")
        return y_mix

    print(f"  [WPE] Raw STFT Shape: {Y.shape}")

    # 2. Validate & Transpose STFT Shape for WPE
    # WPE Core expects: (Frequency, Channels, Time)
    # We need to find which axis is Frequency (257 bins) and which is Channels (2)
    
    freq_bins = N_FFT // 2 + 1  # 257
    num_channels = y_mix.shape[0] # 2
    
    # Detect Frequency Axis
    if Y.shape[0] == freq_bins:
        # (Freq, Ch, Time) - Already correct
        Y_wpe = Y
        transpose_back = None
        print("  [WPE] Shape is already (Freq, Ch, Time).")
        
    elif Y.shape[1] == freq_bins:
        # (Ch, Freq, Time) -> Transpose to (Freq, Ch, Time)
        # Permutation: (1, 0, 2)
        print("  [WPE] Transposing from (Ch, Freq, Time) to (Freq, Ch, Time)...")
        Y_wpe = Y.transpose(1, 0, 2)
        transpose_back = (1, 0, 2) # Inverse is same for (1,0,2)
        
    elif Y.shape[2] == freq_bins:
        # (Ch, Time, Freq) -> Transpose to (Freq, Ch, Time)
        # Permutation: (2, 0, 1)
        print("  [WPE] Transposing from (Ch, Time, Freq) to (Freq, Ch, Time)...")
        Y_wpe = Y.transpose(2, 0, 1)
        transpose_back = (1, 2, 0) # Inverse to get back to (Ch, Time, Freq)
        
    else:
        print(f"  [Error] Could not identify frequency axis (Expected {freq_bins} bins). Shape: {Y.shape}")
        return y_mix

    # 3. Run WPE Optimization
    wpe_opts = dict(taps=taps, delay=delay, iterations=iterations, psd_context=0)
    try:
        Z_wpe = wpe(Y_wpe, **wpe_opts)
    except Exception as e:
        print(f"  [Error] WPE Optimization failed: {e}")
        return y_mix
    
    # 4. Inverse Transpose (Restore shape for ISTFT)
    if transpose_back:
        Z = Z_wpe.transpose(*transpose_back)
    else:
        Z = Z_wpe
        
    # 5. Inverse STFT
    y_dereverb = istft(Z, size=stft_opts['size'], shift=stft_opts['shift'])
    
    # 6. Length Matching
    target_len = y_mix.shape[1]
    
    # Check if ISTFT output dimensions need flipping (just in case)
    if y_dereverb.shape[0] > y_dereverb.shape[1] and y_dereverb.shape[1] == num_channels:
        y_dereverb = y_dereverb.T
        
    if y_dereverb.shape[1] > target_len:
        y_dereverb = y_dereverb[:, :target_len]
    elif y_dereverb.shape[1] < target_len:
        padding = target_len - y_dereverb.shape[1]
        y_dereverb = np.pad(y_dereverb, ((0,0), (0, padding)))
        
    print(f"  [WPE] Output Time-Domain Shape: {y_dereverb.shape}")
    return y_dereverb

def main(args):
    outdir = args.outdir
    
    if outdir is None:
        print("Error: No simulation directory found.")
        return

    print(f"--- Running Standalone WPE Dereverberation ---")
    print(f"Target Directory: {os.path.basename(outdir)}")
    
    mix_path = os.path.join(outdir, "mixture.wav")
    if not os.path.exists(mix_path):
        print(f"Error: {mix_path} not found.")
        return
        
    # 1. Load Mixture
    y_mix, fs = sf.read(mix_path, dtype='float32')
    
    # 2. Transpose to (Channels, Samples) for processing
    if y_mix.ndim > 1 and y_mix.shape[0] > y_mix.shape[1]:
        print(f"  [Load] Transposing input from {y_mix.shape} to (Channels, Samples)")
        y_mix = y_mix.T
    
    # 3. Apply WPE
    y_clean = apply_wpe(y_mix, taps=args.taps, delay=args.delay, iterations=args.iters)
    
    # 4. Save Output
    out_path = os.path.join(outdir, "mixture_wpe.wav")
    sf.write(out_path, y_clean.T, fs)
    
    print(f"Success! Saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default=DEFAULT_OUTDIR)
    parser.add_argument('--taps', type=int, default=10)
    parser.add_argument('--delay', type=int, default=3)
    parser.add_argument('--iters', type=int, default=3)
    
    args = parser.parse_args()
    main(args)