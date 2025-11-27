import numpy as np
import soundfile as sf
import os
import sys

def calculate_metrics_manual(output_signal, target_ref, interf_ref):
    """
    Manually calculates SIR and SDR using signal projection.
    """
    # 1. Normalize all signals to unit energy for fair comparison
    eps = 1e-10
    output_signal = output_signal / (np.linalg.norm(output_signal) + eps)
    target_ref = target_ref / (np.linalg.norm(target_ref) + eps)
    interf_ref = interf_ref / (np.linalg.norm(interf_ref) + eps)

    # 2. Project Output onto Target (How much Target is in the output?)
    # Coefficient alpha = (y . t) / (t . t) -> since normalized, just dot product
    alpha = np.dot(output_signal, target_ref)
    e_target = alpha * target_ref
    
    # 3. Project Output onto Interference (How much Noise is in the output?)
    beta = np.dot(output_signal, interf_ref)
    e_interf = beta * interf_ref
    
    # 4. The rest is Artifacts/Distortion
    e_artif = output_signal - e_target - e_interf
    
    # 5. Calculate Ratios (in dB)
    power_target = np.sum(e_target**2)
    power_interf = np.sum(e_interf**2) + 1e-10 # Avoid div/0
    power_noise  = np.sum(e_artif**2) + 1e-10
    
    sir = 10 * np.log10(power_target / power_interf)
    sdr = 10 * np.log10(power_target / (power_interf + power_noise))
    
    return sdr, sir

def main(output_dir_world):
   
    # 1. Pipeline Integration: Check Path
    if not output_dir_world or not os.path.exists(output_dir_world):
        print(f"ERROR: Invalid directory: {output_dir_world}")
        return

    # 2. Locate Files
    # Inputs are in World_Outputs (the directory passed)
    file_mix = os.path.join(output_dir_world, "mixture_3_sources.wav")
    file_ref_tgt = os.path.join(output_dir_world, "target_reference.wav")
    file_ref_int = os.path.join(output_dir_world, "interference_reference.wav")
    
    # The Output is in MVDR_Outputs (sibling folder to World_Outputs)
    run_root = os.path.dirname(output_dir_world)
    mvdr_dir = os.path.join(run_root, "MVDR_Outputs")
    file_masked = os.path.join(mvdr_dir, "output_masked_mvdr.wav")
    
    if not os.path.exists(file_ref_int):
        print("Error: 'interference_reference.wav' not found. RUN WORLD.PY again.")
        return
    if not os.path.exists(file_masked):
        print(f"Error: MVDR Output not found at: {file_masked}")
        return

    # 3. Load
    s_est, _ = sf.read(file_masked, dtype='float32')
    s_tgt, _ = sf.read(file_ref_tgt, dtype='float32')
    s_int, _ = sf.read(file_ref_int, dtype='float32')
    s_mix, _ = sf.read(file_mix, dtype='float32')
    
    if len(s_mix.shape) > 1: s_mix = s_mix[:, 0] # Mic 1 Baseline

    # 4. Align
    min_len = min(len(s_est), len(s_tgt), len(s_int), len(s_mix))
    s_est = s_est[:min_len]
    s_tgt = s_tgt[:min_len]
    s_int = s_int[:min_len]
    s_mix = s_mix[:min_len]

    # 5. Calculate
    sdr_b, sir_b = calculate_metrics_manual(s_mix, s_tgt, s_int)
    sdr_m, sir_m = calculate_metrics_manual(s_est, s_tgt, s_int)

    # 6. Report to Terminal
    print("\n=== RESULTS ===")
    print(f"Run: {os.path.basename(run_root)}")
    print("-" * 20)
    print(f"BASELINE (Mic 1 Raw):")
    print(f"  SIR: {sir_b:.2f} dB")
    print(f"  SDR: {sdr_b:.2f} dB")
    print("-" * 20)
    print(f"MASKED MVDR:")
    print(f"  SIR: {sir_m:.2f} dB")
    print(f"  SDR: {sdr_m:.2f} dB")
    print("=" * 20)
    
    print(f"SIR IMPROVEMENT: {sir_m - sir_b:.2f} dB")

if __name__ == "__main__":
    # --- CRITICAL FIX: Read the argument passed by the main pipeline script ---
    if len(sys.argv) < 2:
        print("Usage: python run_metrics.py <simulation_output_directory>")
    else:
        main(sys.argv[1])