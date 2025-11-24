import os
import sys
import subprocess
import json
import pandas as pd
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pesq import pesq
import argparse

# Try importing STOI, handle gracefully if missing
try:
    from pystoi import stoi
except ImportError:
    stoi = None

# --- Configuration ---
PYTHON_EXE = sys.executable 
FS = 16000
NUM_ITERATIONS = 500       # Set desired number of runs
SCENARIO = "world"        # <--- UPDATED: "world" (3 sources) is now the default

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
MODEL_PATH = os.path.join(BASE_DIR, "mask_3.pth") 

def ensure_dirs():
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR)

def calc_metrics(est_path, tgt_path, int_path, mix_path):
    """Calculates SIR, PESQ, and STOI."""
    if not os.path.exists(est_path): 
        return {} # File missing, pipeline likely failed
    
    try:
        # Load Files
        est, _ = sf.read(est_path)
        tgt, _ = sf.read(tgt_path)
        intf, _ = sf.read(int_path)
        mix, _ = sf.read(mix_path)

        # Handle formatting (Channels, Length)
        if est.ndim > 1: est = est[:, 0]
        if tgt.ndim > 1: tgt = tgt[:, 0]
        if intf.ndim > 1: intf = intf[:, 0]
        if mix.ndim > 1: mix = mix[:, 0]

        # Truncate to minimum length
        L = min(len(est), len(tgt), len(intf), len(mix))
        est, tgt, intf, mix = est[:L], tgt[:L], intf[:L], mix[:L]

        # --- SIR Calculation ---
        def get_sir_db(y, s, n):
            y = y / (np.linalg.norm(y) + 1e-10)
            s = s / (np.linalg.norm(s) + 1e-10)
            n = n / (np.linalg.norm(n) + 1e-10)
            
            # Project onto subspace
            alpha = np.dot(y, s)
            beta = np.dot(y, n)
            e_tgt = alpha * s
            e_int = beta * n
            
            P_t = np.sum(e_tgt**2)
            P_i = np.sum(e_int**2)
            return 10 * np.log10(P_t / (P_i + 1e-10))

        base_sir = get_sir_db(mix, tgt, intf)
        out_sir = get_sir_db(est, tgt, intf)
        
        # --- PESQ ---
        try: pesq_score = pesq(FS, tgt, est, 'wb')
        except: pesq_score = 0.0
        
        # --- STOI ---
        stoi_score = 0.0
        if stoi:
            try: stoi_score = stoi(tgt, est, FS, extended=False)
            except: pass

        return {
            "Input_SIR": base_sir,
            "Output_SIR": out_sir,
            "PESQ": pesq_score,
            "STOI": stoi_score
        }
    except Exception as e:
        # print(f"Error calculating metrics for {est_path}: {e}")
        return {}

def load_stats(json_path):
    """Loads Time/RAM stats from the sidecar JSON file."""
    if not os.path.exists(json_path): return {}
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except: return {}

def main():
    ensure_dirs()
    print(f"--- Starting Benchmark: {SCENARIO.upper()} ({NUM_ITERATIONS} runs) ---")
    print(f"--- Model Path: {MODEL_PATH} ---")
    
    if not os.path.exists(MODEL_PATH):
        print("WARNING: mask_3.pth not found. The MVDR (Neural) pipeline results will be empty.")
    
    all_results = []

    for i in tqdm(range(NUM_ITERATIONS)):
        row = {"Index": i}
        idx_str = f"{i:02d}"
        
        # --- Define Filenames (Standardized) ---
        f_mix = os.path.join(SAMPLES_DIR, f"{idx_str}_mixture.wav")
        f_tgt = os.path.join(SAMPLES_DIR, f"{idx_str}_target_reference.wav")
        f_int = os.path.join(SAMPLES_DIR, f"{idx_str}_interference_reference.wav")
        
        out_oracle = os.path.join(SAMPLES_DIR, f"{idx_str}_output_oracle.wav")
        out_duet = os.path.join(SAMPLES_DIR, f"{idx_str}_output_duet.wav")
        out_mvdr = os.path.join(SAMPLES_DIR, f"{idx_str}_output_mvdr.wav")
        
        stats_oracle = os.path.join(SAMPLES_DIR, f"{idx_str}_stats_oracle.json")
        stats_duet = os.path.join(SAMPLES_DIR, f"{idx_str}_stats_duet.json")
        stats_mvdr = os.path.join(SAMPLES_DIR, f"{idx_str}_stats_mvdr.json")

        # --- 1. GENERATE DATA (using world.py) ---
        # Note: Ensure world.py accepts --output_dir and --index
        subprocess.run([PYTHON_EXE, "world.py", "--output_dir", SAMPLES_DIR, "--index", idx_str],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Sanity Check: Did generation succeed?
        if not os.path.exists(f_mix):
            print(f"\n[Error] Generation failed for index {i}. 'world.py' did not create {f_mix}.")
            print("Ensure 'world.py' is the refactored version that accepts --output_dir arguments.")
            continue

        # --- 2. RUN PIPELINES ---
        
        # A. Oracle Pipeline
        # Ensure you have renamed oracle_debug.py to oracle.py
        if os.path.exists("oracle.py"):
            subprocess.run([PYTHON_EXE, "oracle.py", 
                            "--mix_path", f_mix, "--ref_tgt", f_tgt, "--ref_int", f_int,
                            "--output_path", out_oracle, "--stats_path", stats_oracle],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
             print("\n[Error] 'oracle.py' not found. Rename oracle_debug.py to oracle.py.")

        # B. DUET Pipeline
        subprocess.run([PYTHON_EXE, "duet.py", 
                        "--mix_path", f_mix, 
                        "--output_path", out_duet, "--stats_path", stats_duet],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # C. MVDR (Neural) Pipeline
        if os.path.exists(MODEL_PATH):
            subprocess.run([PYTHON_EXE, "mvdr.py", 
                            "--mix_path", f_mix, 
                            "--output_path", out_mvdr, 
                            "--model_path", MODEL_PATH, 
                            "--stats_path", stats_mvdr],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # --- 3. AGGREGATE RESULTS ---
        
        # Base Metrics (Input SIR)
        base_metrics = calc_metrics(f_mix, f_tgt, f_int, f_mix)
        row["Input_SIR"] = base_metrics.get("Input_SIR", 0.0)

        # Helper to map raw data to schema
        def add_pipeline_stats(pipeline_suffix, out_path, stat_path):
            m = calc_metrics(out_path, f_tgt, f_int, f_mix)
            s = load_stats(stat_path)
            
            row[f"SIR_{pipeline_suffix}"] = m.get("Output_SIR", np.nan)
            row[f"PESQ_{pipeline_suffix}"] = m.get("PESQ", np.nan)
            row[f"STOI_{pipeline_suffix}"] = m.get("STOI", np.nan)
            row[f"Memory_{pipeline_suffix}"] = s.get("peak_ram_mb", np.nan)
            row[f"Runtime_{pipeline_suffix}"] = s.get("duration_sec", np.nan)

        add_pipeline_stats("oracle", out_oracle, stats_oracle)
        add_pipeline_stats("duet", out_duet, stats_duet)
        add_pipeline_stats("mvdr", out_mvdr, stats_mvdr)

        all_results.append(row)

    # --- 4. SAVE FINAL REPORT ---
    # Define Column Order
    cols = ["Index", "Input_SIR",
            "SIR_oracle", "SIR_duet", "SIR_mvdr",
            "PESQ_oracle", "PESQ_duet", "PESQ_mvdr",
            "STOI_oracle", "STOI_duet", "STOI_mvdr",
            "Memory_oracle", "Memory_duet", "Memory_mvdr",
            "Runtime_oracle", "Runtime_duet", "Runtime_mvdr"]
            
    df = pd.DataFrame(all_results)
    # Filter valid columns only
    df = df[[c for c in cols if c in df.columns]]
    
    csv_name = f"benchmark_results_{SCENARIO}.csv"
    df.to_csv(csv_name, index=False)
    
    print("\n" + "="*60)
    print(f"BENCHMARK COMPLETE. Results saved to: {csv_name}")
    print("="*60)
    
    # --- 5. PRINT SUMMARY ---
    print("\n--- Averages ---")
    print(df.mean().to_string())

if __name__ == "__main__":
    main()