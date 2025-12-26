import librosa
import numpy as np
import os
import sys

def load_and_preprocess(file_path, target_sr=22050):
    """
    Loads audio and standardizes it:
    1. Resamples to target_sr (to ensure consistency).
    2. Converts to mono.
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def calculate_waveform_correlation(y1, y2):
    """
    Calculates the Pearson correlation between two waveforms.
    Returns: Score between -1 and 1. (1.0 is a perfect match).
    """
    # Truncate to the length of the shorter audio to compare
    min_len = min(len(y1), len(y2))
    y1_trim = y1[:min_len]
    y2_trim = y2[:min_len]
    
    # Pearson Correlation Coefficient
    correlation = np.corrcoef(y1_trim, y2_trim)[0, 1]
    return correlation

def calculate_spectral_similarity(y1, y2, sr):
    """
    Calculates similarity based on MFCCs (Mel-frequency cepstral coefficients).
    Returns: A 'Distance' score (Lower is better).
    """
    # Extract features (MFCCs)
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13)

    # Align frames (truncate to shorter)
    min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1 = mfcc1[:, :min_frames]
    mfcc2 = mfcc2[:, :min_frames]

    # Calculate Euclidean distance between the features
    dist = np.linalg.norm(mfcc1 - mfcc2) / min_frames
    return dist

def compare_direct_files(file1_path, file2_path):
    print(f"--- Loading Files ---")
    print(f"Target:    {os.path.basename(file1_path)}")
    print(f"Candidate: {os.path.basename(file2_path)}")
    
    # Load both files
    y1, sr1 = load_and_preprocess(file1_path)
    y2, sr2 = load_and_preprocess(file2_path)
    
    if y1 is None or y2 is None:
        print("Error: Could not load one or both audio files.")
        return

    # Calculate Metrics
    corr = calculate_waveform_correlation(y1, y2)
    spec_dist = calculate_spectral_similarity(y1, y2, sr1)

    # Print Results
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-" * 50)
    
    # Context for Correlation
    corr_desc = "(1.0 = Exact Match)" if corr > 0.99 else "(0.0 = No Correlation)"
    print(f"{'Waveform Correlation':<30} | {corr:.4f} {corr_desc}")

    # Context for Spectral Distance
    spec_desc = "(Lower is Better)"
    print(f"{'Spectral Distance (MFCC)':<30} | {spec_dist:.4f} {spec_desc}")
    print("="*50)

# --- CONFIGURATION ---
# Replace these with your direct file paths
TARGET_FILE = "model training/debug_audio/epoch_001/target_audio.wav"
COMPARE_FILE = "model training/debug_audio/epoch_001/mask_truth_audio.wav"

if __name__ == "__main__":
    # Check if files exist before running
    if os.path.exists(TARGET_FILE) and os.path.exists(COMPARE_FILE):
        compare_direct_files(TARGET_FILE, COMPARE_FILE)
    else:
        print(f"Error: One or more files not found at paths specified:\n{TARGET_FILE}\n{COMPARE_FILE}")