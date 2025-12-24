import numpy as np
import soundfile as sf
import os
import csv
import argparse
from scipy.signal import resample

# --- CONFIGURATION ---
RESULTS_CSV = "evaluation_results.csv"

# Handle optional dependencies
try:
    from pesq import pesq
    PESQ_INSTALLED = True
except ImportError:
    print("Warning: 'pesq' not installed. PESQ scores will be 0.")
    PESQ_INSTALLED = False

def append_to_csv(filename, row_dict):
    """Appends a dictionary of metrics to a CSV file."""
    file_exists = os.path.isfile(filename)
    # standard headers
    headers = ["Filename", "SI_SDR", "SNR", "PESQ_WB", "PESQ_NB"]
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def calculate_si_sdr(reference, estimate):
    """
    Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    This serves as the primary 'Signal to Noise' metric when 
    interference files are not available.
    """
    eps = 1e-10
    
    # Ensure zero-mean
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Calculate optimal scaling factor (alpha) to match reference
    # formula: alpha = <x, s> / <s, s>
    dot_prod = np.dot(estimate, reference)
    ref_energy = np.dot(reference, reference) + eps
    alpha = dot_prod / ref_energy
    
    # Projection
    e_target = alpha * reference
    e_noise = estimate - e_target # This contains both noise and artifacts
    
    # Energies
    target_pow = np.sum(e_target**2) + eps
    noise_pow = np.sum(e_noise**2) + eps
    
    si_sdr = 10 * np.log10(target_pow / noise_pow)
    return si_sdr

def calculate_snr(reference, estimate):
    """Calculates standard SNR."""
    eps = 1e-10
    noise = reference - estimate
    
    ref_pow = np.sum(reference**2) + eps
    noise_pow = np.sum(noise**2) + eps
    
    return 10 * np.log10(ref_pow / noise_pow)

def evaluate_pair(ref_path, est_path):
    """Loads files, aligns them, and computes metrics."""
    try:
        # Load Files
        ref_audio, fs_ref = sf.read(ref_path, dtype='float32')
        est_audio, fs_est = sf.read(est_path, dtype='float32')
        
        # Check Samplerate mismatch
        if fs_ref != fs_est:
            print(f"[WARN] Samplerate mismatch: Ref {fs_ref} vs Est {fs_est}. Skipping.")
            return None

        # Handle Multichannel (Take Channel 0)
        if ref_audio.ndim > 1: ref_audio = ref_audio[:, 0]
        if est_audio.ndim > 1: est_audio = est_audio[:, 0]
        
        # Align lengths (truncate to minimum)
        min_len = min(len(ref_audio), len(est_audio))
        ref_audio = ref_audio[:min_len]
        est_audio = est_audio[:min_len]
        
        # 1. Compute SI-SDR (The 2-file alternative to SIR)
        si_sdr = calculate_si_sdr(ref_audio, est_audio)
        snr = calculate_snr(ref_audio, est_audio)
        
        # 2. Compute PESQ
        pesq_nb = 0.0
        pesq_wb = 0.0
        
        if PESQ_INSTALLED:
            try:
                # PESQ requires specific sample rates (8k or 16k)
                curr_ref = ref_audio
                curr_est = est_audio
                curr_fs = fs_ref

                # Resample to 16k if necessary for PESQ
                if curr_fs not in [8000, 16000]:
                    target_fs = 16000
                    num_samples = int(len(curr_ref) * target_fs / curr_fs)
                    curr_ref = resample(curr_ref, num_samples)
                    curr_est = resample(curr_est, num_samples)
                    curr_fs = target_fs

                if curr_fs == 16000:
                    pesq_wb = pesq(curr_fs, curr_ref, curr_est, 'wb')
                    pesq_nb = pesq(curr_fs, curr_ref, curr_est, 'nb')
                elif curr_fs == 8000:
                    pesq_nb = pesq(curr_fs, curr_ref, curr_est, 'nb')
                    
            except Exception as e:
                print(f"PESQ Calculation Error: {e}")

        # Return results
        return {
            "Filename": os.path.basename(est_path),
            "SI_SDR": f"{si_sdr:.2f}",
            "SNR": f"{snr:.2f}",
            "PESQ_WB": f"{pesq_wb:.4f}",
            "PESQ_NB": f"{pesq_nb:.4f}"
        }

    except Exception as e:
        print(f"Error processing {est_path}: {e}")
        return None

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Example Usage: Replace these paths with your actual folders/files
    # For single file mode, you can just set these variables manually
    
    # OPTION 1: Hardcode paths here
    REF_FILE = "/home/rpzrm/global/projects/real-time-audio-visual-zooming/Final_pipeline/data/simulated/test1/mixture.wav" 
    EST_FILE = "/home/rpzrm/global/projects/real-time-audio-visual-zooming/Final_pipeline/data/results/test1_results/test1_enhanced.wav"
    
    print(f"--- Evaluating Single Pair ---")
    if os.path.exists(REF_FILE) and os.path.exists(EST_FILE):
        metrics = evaluate_pair(REF_FILE, EST_FILE)
        if metrics:
            print("Results:")
            print(f"  SI-SDR:  {metrics['SI_SDR']} dB")
            print(f"  PESQ WB: {metrics['PESQ_WB']}")
            
            # Save to CSV
            append_to_csv(RESULTS_CSV, metrics)
            print(f"Saved to {RESULTS_CSV}")
    else:
        print("Please set valid file paths in the script to run.")