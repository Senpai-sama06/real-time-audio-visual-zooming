import numpy as np
import soundfile as sf
import os

def calculate_metrics_manual(output_signal, target_ref, interf_ref):
    """
    Manually calculates SIR and SDR.
    This avoids the 'inf' issues by doing a simple projection.
    """
    # 1. Normalize all signals to unit energy for fair comparison
    output_signal = output_signal / np.linalg.norm(output_signal)
    target_ref = target_ref / np.linalg.norm(target_ref)
    interf_ref = interf_ref / np.linalg.norm(interf_ref)

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

def main():
    print("--- 3. Validation (Manual Projection) ---")
    
    # 1. Files
    file_mix = "mixture_3_sources.wav"
    file_masked = "output_oracle.wav"
    file_ref_tgt = "target_reference.wav"
    file_ref_int = "interference_reference.wav"
    
    if not os.path.exists(file_ref_int):
        print("Error: 'interference_reference.wav' not found. RUN WORLD.PY AGAIN.")
        return

    # 2. Load
    s_est, _ = sf.read(file_masked, dtype='float32')
    s_tgt, _ = sf.read(file_ref_tgt, dtype='float32')
    s_int, _ = sf.read(file_ref_int, dtype='float32')
    s_mix, _ = sf.read(file_mix, dtype='float32')
    if len(s_mix.shape) > 1: s_mix = s_mix[:, 0] # Mic 1 Baseline

    # 3. Align
    min_len = min(len(s_est), len(s_tgt), len(s_int), len(s_mix))
    s_est = s_est[:min_len]
    s_tgt = s_tgt[:min_len]
    s_int = s_int[:min_len]
    s_mix = s_mix[:min_len]

    # 4. Calculate
    sdr_b, sir_b = calculate_metrics_manual(s_mix, s_tgt, s_int)
    sdr_m, sir_m = calculate_metrics_manual(s_est, s_tgt, s_int)

    # 5. Report
    print("\n=== RESULTS ===")
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
    main()