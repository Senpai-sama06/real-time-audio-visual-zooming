import numpy as np
import soundfile as sf
import os
import datetime 
from src import config
import csv
# Handle optional dependencies gracefully
try:
    from pystoi import stoi
    from pesq import pesq
    DEPENDENCIES_OK = True
except ImportError:
    print("Warning: 'pystoi' or 'pesq' not installed. Metrics will be skipped.")
    DEPENDENCIES_OK = False

def append_to_csv(run_name, metrics_dict):
    """Appends metrics to a central CSV file."""
    csv_path = os.path.join(config.RESULTS_DIR, "batch_metrics.csv")
    
    # Define headers
    headers = ["Run_ID", "SIR_Base", "SIR_Enh", "SIR_Imp", 
               "SINR_Base", "SINR_Enh", "STOI", "PESQ_WB", "PESQ_NB"]
    
    # Check if file exists to write headers
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        # Prepare row
        row = {
            "Run_ID": run_name,
            "SIR_Base": f"{metrics_dict['sir_b']:.2f}",
            "SIR_Enh": f"{metrics_dict['sir_s']:.2f}",
            "SIR_Imp": f"{metrics_dict['imp']:.2f}",
            "SINR_Base": f"{metrics_dict['sinr_b']:.2f}",
            "SINR_Enh": f"{metrics_dict['sinr_s']:.2f}",
            "STOI": f"{metrics_dict['stoi']:.4f}",
            "PESQ_WB": f"{metrics_dict['pesq_wb']:.4f}",
            "PESQ_NB": f"{metrics_dict['pesq_nb']:.4f}",
        }
        writer.writerow(row)

# --- HELPER CLASSES ---

class PESQEvaluator:
    def __init__(self, ref_audio, deg_audio, fs):
        self.fs = fs
        self.ref_audio = ref_audio
        self.deg_audio = deg_audio

    def evaluate(self):
        nb_score, wb_score = 0.0, 0.0
        if not DEPENDENCIES_OK: return 0.0, 0.0

        try:
            if self.fs in [8000, 16000]:
                nb_score = pesq(self.fs, self.ref_audio, self.deg_audio, 'nb')
            if self.fs == 16000:
                wb_score = pesq(self.fs, self.ref_audio, self.deg_audio, 'wb')
        except Exception as e:
            print(f"PESQ Error: {e}")
            
        return nb_score, wb_score

# --- CORE LOGIC ---

def load_and_align(sim_dir, result_path):
    """
    Loads Ground Truth from sim_dir and Estimate from result_path.
    Aligns them to the minimum length.
    """
    try:
        # Load Estimate
        s_est, _ = sf.read(result_path, dtype='float32')
        
        # Load Ground Truths
        s_tgt, _ = sf.read(os.path.join(sim_dir, "target.wav"), dtype='float32')
        s_int, _ = sf.read(os.path.join(sim_dir, "interference.wav"), dtype='float32')
        s_mix, _ = sf.read(os.path.join(sim_dir, "mixture.wav"), dtype='float32')
        
        # Handle Multichannel (Take Channel 0)
        if s_mix.ndim > 1: s_mix = s_mix[:, 0]
        if s_est.ndim > 1: s_est = s_est[:, 0]
        if s_tgt.ndim > 1: s_tgt = s_tgt[:, 0]
        if s_int.ndim > 1: s_int = s_int[:, 0]

        # Align lengths
        min_len = min(len(s_est), len(s_tgt), len(s_int), len(s_mix))
        return (
            s_est[:min_len].astype(np.float64),
            s_tgt[:min_len].astype(np.float64),
            s_int[:min_len].astype(np.float64),
            s_mix[:min_len].astype(np.float64)
        )
    except FileNotFoundError as e:
        print(f"[EVAL] Error: Missing file - {e}")
        return None, None, None, None

def calculate_osnr_osir(output, target, interferer):
    """Calculates OSINR and OSIR using projection."""
    eps = 1e-10
    # Normalize
    target = target / (np.linalg.norm(target) + eps)
    interferer = interferer / (np.linalg.norm(interferer) + eps)

    # Project
    alpha = np.dot(output, target)
    beta = np.dot(output, interferer)
    
    e_target = alpha * target
    e_interf = beta * interferer
    e_noise = output - e_target - e_interf

    P_t = np.sum(e_target**2)
    P_i = np.sum(e_interf**2)
    P_n = np.sum(e_noise**2)

    OSINR = 10 * np.log10(P_t / (P_i + P_n + eps))
    OSIR = 10 * np.log10(P_t / (P_i + eps))
    return OSINR, OSIR

def evaluate_run(run_name):
    """
    Main entry point. 
    Reads from data/simulated/{run_name} and data/results/{run_name}_results
    Writes report to data/results/{run_name}_results/report.txt
    """
    sim_dir = os.path.join(config.SIM_DIR, run_name)
    res_dir = os.path.join(config.RESULTS_DIR, f"{run_name}_results")
    est_path = os.path.join(res_dir, f"{run_name}_enhanced.wav")
    report_path = os.path.join(res_dir, "report.txt")

    if not os.path.exists(est_path):
        print(f"[EVAL] Error: Inference output not found at {est_path}")
        return

    print(f"[EVAL] Evaluating: {run_name}...")

    # 1. Load Data
    s_est, s_tgt, s_int, s_mix = load_and_align(sim_dir, est_path)
    if s_est is None: return

    # 2. Compute Metrics
    # Baseline (Input Mixture)
    osinr_b, osir_b = calculate_osnr_osir(s_mix, s_tgt, s_int)
    
    # Solution (Enhanced)
    osinr_s, osir_s = calculate_osnr_osir(s_est, s_tgt, s_int)
    
    # Perceptual Metrics
    stoi_score = 0.0
    pesq_wb, pesq_nb = 0.0, 0.0
    
    if DEPENDENCIES_OK:
        stoi_score = stoi(s_tgt, s_est, config.FS, extended=False)
        pesq_eval = PESQEvaluator(s_tgt, s_est, config.FS)
        pesq_nb, pesq_wb = pesq_eval.evaluate()

    # 3. Generate Report
    improvement = osir_s - osir_b
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = [
        f"=== EVALUATION REPORT: {run_name} ===",
        f"Date: {timestamp}",
        f"------------------------------------",
        f"BASELINE (Mixture):",
        f"  SIR:   {osir_b:.2f} dB",
        f"  SINR:  {osinr_b:.2f} dB",
        f"------------------------------------",
        f"ENHANCED (Output):",
        f"  SIR:   {osir_s:.2f} dB",
        f"  SINR:  {osinr_s:.2f} dB",
        f"  STOI:  {stoi_score:.4f}",
        f"  PESQ:  {pesq_wb:.4f} (WB) | {pesq_nb:.4f} (NB)",
        f"------------------------------------",
        f"SIR IMPROVEMENT: {improvement:+.2f} dB",
        f"===================================="
    ]
    
    report_str = "\n".join(report)
    print(report_str)
    
    with open(report_path, "w") as f:
        f.write(report_str)
    print(f"[EVAL] Report saved to: {report_path}")
    
    if DEPENDENCIES_OK:
        stoi_score = stoi(s_tgt, s_est, config.FS, extended=False)
        pesq_eval = PESQEvaluator(s_tgt, s_est, config.FS)
        pesq_nb, pesq_wb = pesq_eval.evaluate()

    improvement = osir_s - osir_b

    # --- NEW: PACK DATA ---
    metrics_data = {
        "sir_b": osir_b, "sir_s": osir_s, "imp": improvement,
        "sinr_b": osinr_b, "sinr_s": osinr_s,
        "stoi": stoi_score, "pesq_wb": pesq_wb, "pesq_nb": pesq_nb
    }
    
    # Save to CSV
    append_to_csv(run_name, metrics_data)