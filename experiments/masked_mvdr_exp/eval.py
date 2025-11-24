import numpy as np
import soundfile as sf
import os
import sys
import datetime 
from typing import Union, Tuple

# External Libraries required for Objective Metrics
try:
    from pystoi import stoi
    from pesq import pesq
except ImportError:
    print("FATAL ERROR: Required packages (pystoi/pesq) not found. Please install them.")
    sys.exit(1)


# --- PESQ EVALUATOR CLASS ---

class PESQEvaluator:
    """
    An object-oriented class for calculating the Perceptual Evaluation of Speech
    Quality (PESQ) score between a reference and a degraded audio signal.
    """
    def __init__(self, ref_audio: np.ndarray, deg_audio: np.ndarray, fs: int):
        """
        Initializes the evaluator with audio data.
        """
        self.fs = fs
        self.ref_audio = ref_audio
        self.deg_audio = deg_audio

    def calculate_pesq(self, mode: str) -> float:
        """
        Calculates the PESQ score for a given mode.
        """
        if self.fs is None or self.ref_audio is None or self.deg_audio is None:
            raise RuntimeError("Audio data was not loaded correctly.")

        # Allow 16000 Hz for Narrow-Band, as the pesq library handles downsampling.
        if mode == 'nb' and self.fs not in [8000, 16000]:
            raise ValueError(
                f"Narrow-Band PESQ requires 8000 Hz or 16000 Hz input, but audio is {self.fs} Hz."
            )
        # Wide-Band requires exactly 16000 Hz.
        elif mode == 'wb' and self.fs != 16000:
            raise ValueError(
                f"Wide-Band PESQ requires 16000 Hz, but audio is {self.fs} Hz."
            )
        elif mode not in ['nb', 'wb']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'nb' or 'wb'.")

        score = pesq(self.fs, self.ref_audio, self.deg_audio, mode)
        return score

    def evaluate(self) -> Tuple[float, float]:
        """
        Runs both Narrow-Band and Wide-Band evaluation, if supported by the fs.

        Returns:
            A tuple (nb_pesq_score, wb_pesq_score). Scores will be 0.0 if not supported.
        """
        nb_score = 0.0
        wb_score = 0.0

        if self.fs == 16000:
            nb_score = self.calculate_pesq('nb')
            wb_score = self.calculate_pesq('wb')
        elif self.fs == 8000:
            nb_score = self.calculate_pesq('nb')
            
        return nb_score, wb_score

# --- CONFIGURATION ---
FS = 16000 
HISTORY_FILE = "evaluation_history.txt"


def load_and_align_signals(output_file_full_path, output_path):
    """Loads all required files and aligns them to the minimum length."""
    
    try:
        # Load the estimated signal
        s_est, _ = sf.read(output_file_full_path, dtype='float32')
        
        # Load reference files using the output_path
        s_tgt_ref, _ = sf.read(os.path.join(output_path, "target_reference.wav"), dtype='float32')
        s_int_ref, _ = sf.read(os.path.join(output_path, "interference_reference.wav"), dtype='float32')
        s_mix, _ = sf.read(os.path.join(output_path, "mixture_3_sources.wav"), dtype='float32')
        
    except FileNotFoundError as e:
        missing_file = str(e).split("'")[1]
        print(f"Error loading files: Missing file '{missing_file}'. Ensure all files exist in the specified path.")
        return None, None, None, None, None

    # Handle multi-channel (keep only first channel)
    if len(s_mix.shape) > 1: s_mix = s_mix[:, 0]
    if len(s_est.shape) > 1: s_est = s_est[:, 0]
    if len(s_tgt_ref.shape) > 1: s_tgt_ref = s_tgt_ref[:, 0]
    if len(s_int_ref.shape) > 1: s_int_ref = s_int_ref[:, 0]

    # Align to minimum length
    min_len = min(len(s_est), len(s_tgt_ref), len(s_int_ref), len(s_mix))
    
    # Cast to float64 for PESQ compatibility
    s_est = s_est[:min_len].astype(np.float64)
    s_tgt_ref = s_tgt_ref[:min_len].astype(np.float64)
    s_int_ref = s_int_ref[:min_len].astype(np.float64)
    s_mix = s_mix[:min_len].astype(np.float64)
    
    return s_est, s_tgt_ref, s_int_ref, s_mix, min_len

def calculate_osnr_and_osir(output_signal, target_ref, interf_ref):
    """Calculates OSINR and OSIR using projection method."""
    
    floor_norm = 1e-10
    
    # 1. Normalize all inputs to unit energy
    target_ref = target_ref / (np.linalg.norm(target_ref) + floor_norm)
    interf_ref = interf_ref / (np.linalg.norm(interf_ref) + floor_norm)

    # 2. Projection Method
    alpha = np.dot(output_signal, target_ref)
    e_target = alpha * target_ref
    
    beta = np.dot(output_signal, interf_ref)
    e_interf = beta * interf_ref
    
    # 3. Calculate Residual Error (Artifacts/Noise)
    e_artif_noise = output_signal - e_target - e_interf
    
    # 4. Calculate Powers
    P_target = np.sum(e_target**2)
    P_interf = np.sum(e_interf**2)
    P_noise  = np.sum(e_artif_noise**2)
    
    floor_log = 1e-10           
    
    # OSINR = 10 * log10( P_target / (P_interf + P_noise) )
    OSINR = 10 * np.log10( P_target / (P_interf + P_noise + floor_log) )
    
    # OSIR = 10 * log10( P_target / P_interf )
    OSIR = 10 * np.log10( P_target / (P_interf + floor_log) )
    
    return OSINR, OSIR, P_target, P_interf, P_noise

def calculate_pesq_metric(target_ref, processed_signal, fs):
    """Calculates PESQ Wide-Band and Narrow-Band scores."""
    try:
        evaluator = PESQEvaluator(target_ref, processed_signal, fs)
        nb_score, wb_score = evaluator.evaluate()
        return wb_score, nb_score
    except Exception as e:
        print(f"Warning: PESQ calculation failed: {e}")
        return 0.0, 0.0 # Return 0.0 if calculation fails


def main():
    print("--- SP CUP 2026: Official Metrics Scoreboard ---")
    
    # --- HARDCODED PATH DEFINITIONS (Modify these for your setup) ---
    OUTPUT_PATH = "/home/cse-sdpl/paarth/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/samples"
    # OUTPUT_PATH = "/home/rpzrm/global/projects/real-time-audio-visual-zooming/experiments/masked_mvdr_exp/samples"
    # BASE_FILENAME = "output_neural_mvdr"
    BASE_FILENAME = "output_oracle"
    # BASE_FILENAME = "output_neural_zoom_5"
    
    # Construct the full path for the estimated signal
    output_file_full_path = os.path.join(OUTPUT_PATH, f"{BASE_FILENAME}.wav")
    
    # Load all signals (s_est = estimated, s_tgt = target, s_int = interference, s_mix = mixture)
    s_est, s_tgt, s_int, s_mix, L = load_and_align_signals(output_file_full_path, OUTPUT_PATH)
    
    if L is None: return

    # --- 1. Calculate Metrics ---
    
    # Calculate baseline metrics for comparison (s_mix)
    OSINR_b, OSIR_b, _, _, _ = calculate_osnr_and_osir(s_mix, s_tgt, s_int)

    # Calculate solution metrics (s_est)
    OSINR_s, OSIR_s, P_target_s, P_interf_s, P_noise_s = calculate_osnr_and_osir(s_est, s_tgt, s_int)
    
    # STOI
    STOI_score = stoi(s_tgt, s_est, FS)
    
    # PESQ
    PESQ_WB_score, PESQ_NB_score = calculate_pesq_metric(s_tgt, s_est, FS)
    
    # --- 2. FORMAT OUTPUT ---
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SIR_IMPROVEMENT = OSIR_s - OSIR_b
    
    report_lines = [
        f"========================================================",
        f"EVALUATION RUNTIME: {current_time}",
        f"FILE TESTED:        {os.path.basename(output_file_full_path)}",
        f"--------------------------------------------------------",
        f"BASELINE (Raw Mix):",
        f"  Input SIR: {OSIR_b:.2f} dB",
        f"  Input OSINR: {OSINR_b:.2f} dB",
        f"--------------------------------------------------------",
        f"PROCESSED SIGNAL:",
        f"  Output OSIR: {OSIR_s:.2f} dB",
        f"  Output OSINR: {OSINR_s:.2f} dB",
        f"  STOI Score:  {STOI_score:.4f}",
        f"  PESQ-WB Score: {PESQ_WB_score:.4f}",
        f"  PESQ-NB Score: {PESQ_NB_score:.4f}",
        f"--------------------------------------------------------",
        f"TOTAL SIR IMPROVEMENT: +{SIR_IMPROVEMENT:.2f} dB",
        f"========================================================\n"
    ]

    report = "\n".join(report_lines)
    
    # --- 3. APPEND TO HISTORY FILE ---
    try:
        with open(os.path.join(OUTPUT_PATH, HISTORY_FILE), 'a') as f:
            f.write(report)
        print("\nReport successfully generated:")
        print(report)
        print(f"Results appended to {os.path.join(OUTPUT_PATH, HISTORY_FILE)}")
    except Exception as e:
        print(f"\nFATAL ERROR: Could not write to history file. {e}")

if __name__ == "__main__":
    main()