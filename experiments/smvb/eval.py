import numpy as np
import soundfile as sf
import os
import sys
import datetime 
from typing import Union, Tuple

# External Libraries
try:
    from pystoi import stoi
    from pesq import pesq
except ImportError:
    print("FATAL ERROR: Required packages (pystoi/pesq) not found. Please install them.")
    sys.exit(1)


# --- CONFIGURATION ---
FS = 16000 
HISTORY_FILE = "evaluation_history.txt"


# --- PESQ EVALUATOR CLASS (Fixed for Data Types) ---

class PESQEvaluator:
    def __init__(self, ref_audio: np.ndarray, deg_audio: np.ndarray, fs: int):
        self.fs = fs
        self.ref_audio = ref_audio
        self.deg_audio = deg_audio

    def calculate_pesq(self, mode: str) -> float:
        if self.fs is None or self.ref_audio is None or self.deg_audio is None:
            raise RuntimeError("Audio data was not loaded correctly.")

        if mode == 'nb' and self.fs not in [8000, 16000]:
            raise ValueError(f"Narrow-Band PESQ requires 8k/16k Hz, but audio is {self.fs} Hz.")
        elif mode == 'wb' and self.fs != 16000:
            raise ValueError(f"Wide-Band PESQ requires 16000 Hz, but audio is {self.fs} Hz.")

        # FIX: Explicit cast to float32 for the C-wrapper
        ref_32 = self.ref_audio.astype(np.float32)
        deg_32 = self.deg_audio.astype(np.float32)

        score = pesq(self.fs, ref_32, deg_32, mode)
        return score

    def evaluate(self) -> Tuple[float, float]:
        nb_score = 0.0
        wb_score = 0.0

        # Run WB and NB based on Sampling Rate
        if self.fs == 16000:
            nb_score = self.calculate_pesq('nb')
            wb_score = self.calculate_pesq('wb')
        elif self.fs == 8000:
            nb_score = self.calculate_pesq('nb')
            
        return nb_score, wb_score


# --- DATA LOADING (Fixed for Sample Rate Safety) ---

def load_and_align_signals(output_file_full_path, output_path):
    """Loads files, CHECKS FS, and aligns to minimum length."""
    
    files_to_load = {
        "est": output_file_full_path,
        "tgt": os.path.join(output_path, "target.wav"),
        "int": os.path.join(output_path, "interference.wav"),
        "mix": os.path.join(output_path, "mixture.wav")
    }
    
    loaded_signals = {}
    
    try:
        for key, path in files_to_load.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")
            
            data, samplerate = sf.read(path, dtype='float32')
            
            # FIX: Critical Sample Rate Check
            if samplerate != FS:
                print(f"CRITICAL ERROR: File {os.path.basename(path)} is {samplerate}Hz. Expected {FS}Hz.")
                print("Evaluation aborted to prevent incorrect metrics.")
                return None, None, None, None, None
            
            # Handle multi-channel (keep only first channel)
            if len(data.shape) > 1: 
                data = data[:, 0]
                
            loaded_signals[key] = data

    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None, None, None, None

    # Align to minimum length
    min_len = min(len(loaded_signals["est"]), len(loaded_signals["tgt"]), 
                  len(loaded_signals["int"]), len(loaded_signals["mix"]))
    
    # FIX: Warn if large difference in length (indicative of mismatch)
    if len(loaded_signals["est"]) != len(loaded_signals["tgt"]):
        diff = abs(len(loaded_signals["est"]) - len(loaded_signals["tgt"]))
        if diff > FS * 0.5: # More than 0.5 seconds difference
            print(f"WARNING: Length mismatch detected ({diff} samples). Metrics might be affected if latency exists.")

    # Cast to float64 for high-precision math (Si-SDR/OSINR)
    # (Note: PESQ class now handles the downcast to float32 internally)
    s_est = loaded_signals["est"][:min_len].astype(np.float64)
    s_tgt = loaded_signals["tgt"][:min_len].astype(np.float64)
    s_int = loaded_signals["int"][:min_len].astype(np.float64)
    s_mix = loaded_signals["mix"][:min_len].astype(np.float64)
    
    return s_est, s_tgt, s_int, s_mix, min_len


def calculate_sisdr(reference, estimate):
    """Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)."""
    eps = np.finfo(estimate.dtype).eps
    
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    
    # Project estimate onto reference
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    
    e_target = alpha * reference
    e_res = estimate - e_target
    
    numerator = np.sum(e_target ** 2)
    denominator = np.sum(e_res ** 2) + eps
    
    sisdr = 10 * np.log10(numerator / denominator)
    return sisdr


def calculate_osnr_and_osir(output_signal, target_ref, interf_ref):
    """Calculates OSINR and OSIR using projection method."""
    floor_norm = 1e-10
    
    # Normalize to unit energy
    target_ref = target_ref / (np.linalg.norm(target_ref) + floor_norm)
    interf_ref = interf_ref / (np.linalg.norm(interf_ref) + floor_norm)

    # Projection
    alpha = np.dot(output_signal, target_ref)
    e_target = alpha * target_ref
    # (target+interf+noise).target
    
    beta = np.dot(output_signal, interf_ref)
    e_interf = beta * interf_ref
    
    e_artif_noise = output_signal - e_target - e_interf
    
    P_target = np.sum(e_target**2)
    P_interf = np.sum(e_interf**2)
    P_noise  = np.sum(e_artif_noise**2)
    
    floor_log = 1e-10           
    
    OSINR = 10 * np.log10( P_target / (P_interf + P_noise + floor_log) )
    OSIR = 10 * np.log10( P_target / (P_interf + floor_log) )
    
    return OSINR, OSIR, P_target, P_interf, P_noise


def calculate_pesq_metric(target_ref, processed_signal, fs):
    try:
        evaluator = PESQEvaluator(target_ref, processed_signal, fs)
        return evaluator.evaluate()
    except Exception as e:
        print(f"Warning: PESQ calculation failed: {e}")
        return 0.0, 0.0


def main():
    print("--- SP CUP 2026: Official Metrics Scoreboard ---")
    
    # --- PATHS (Modify as needed) ---
    OUTPUT_PATH = "sample"
    BASE_FILENAME = "sample_enhanced"
    
    output_file_full_path = os.path.join(OUTPUT_PATH, f"{BASE_FILENAME}.wav")
    
    # 1. Load Data
    s_est, s_tgt, s_int, s_mix, L = load_and_align_signals(output_file_full_path, OUTPUT_PATH)
    
    if L is None: return # Exit if loading failed

    # 2. Calculate Metrics
    # Baseline
    OSINR_b, OSIR_b, _, _, _ = calculate_osnr_and_osir(s_mix, s_tgt, s_int)

    # Solution
    OSINR_s, OSIR_s, P_target_s, P_interf_s, P_noise_s = calculate_osnr_and_osir(s_est, s_tgt, s_int)
    STOI_score = stoi(s_tgt, s_est, FS)
    PESQ_WB_score, PESQ_NB_score = calculate_pesq_metric(s_tgt, s_est, FS)
    SiSDR_score = calculate_sisdr(s_tgt, s_est)
    
    # 3. Report
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SIR_IMPROVEMENT = OSIR_s - OSIR_b
    
    report_lines = [
        f"========================================================",
        f"EVALUATION RUNTIME: {current_time}",
        f"FILE TESTED:        {os.path.basename(output_file_full_path)}",
        f"--------------------------------------------------------",
        f"BASELINE (Raw Mix):",
        f"  Input SIR:   {OSIR_b:.2f} dB",
        f"  Input OSINR: {OSINR_b:.2f} dB",
        f"--------------------------------------------------------",
        f"PROCESSED SIGNAL:",
        f"  Output OSIR:   {OSIR_s:.2f} dB",
        f"  Output OSINR:  {OSINR_s:.2f} dB",
        f"  STOI Score:    {STOI_score:.4f}",
        f"  PESQ-WB Score: {PESQ_WB_score:.4f}",
        f"  PESQ-NB Score: {PESQ_NB_score:.4f}",
        f"  Si-SDR Score:  {SiSDR_score:.4f} dB",
        f"--------------------------------------------------------",
        f"TOTAL SIR IMPROVEMENT: +{SIR_IMPROVEMENT:.2f} dB",
        f"========================================================\n"
    ]

    report = "\n".join(report_lines)
    
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