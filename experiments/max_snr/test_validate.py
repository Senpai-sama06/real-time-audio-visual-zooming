import numpy as np
import soundfile as sf
import os

def calculate_metrics_manual(output_signal, target_ref, interf_ref):
    """Manually calculates SIR and SDR via projection."""
    
    # Align lengths
    min_len = min(len(output_signal), len(target_ref), len(interf_ref))
    output_signal = output_signal[:min_len]
    target_ref = target_ref[:min_len]
    interf_ref = interf_ref[:min_len]
    
    # 1. Normalize
    output_signal = output_signal / np.linalg.norm(output_signal)
    target_ref = target_ref / np.linalg.norm(target_ref)
    interf_ref = interf_ref / np.linalg.norm(interf_ref)

    # 2. Project
    e_target = np.dot(output_signal, target_ref) * target_ref
    e_interf = np.dot(output_signal, interf_ref) * interf_ref
    e_artif = output_signal - e_target - e_interf
    
    # 3. Ratios
    power_target = np.sum(e_target**2)
    power_interf = np.sum(e_interf**2) + 1e-10
    power_noise  = np.sum(e_artif**2) + 1e-10
    
    sir = 10 * np.log10(power_target / power_interf)
    sdr = 10 * np.log10(power_target / (power_interf + power_noise))
    
    return sdr, sir

def main():
    print("--- 3. Validation (Test Architecture) ---")
    
    # 1. Find the most recent MaxSNR output file
    # This is tricky, let's hardcode for the test
    # You will need to change this if you change the FOV_WIDTH
    
    fov_width_for_test = 20.0 # <-- MATCH THIS TO YOUR test_maxsnr.py
    
    # file_maxsnr = f"output_maxsnr_fov_{fov_width_for_test}deg.wav"
    file_maxsnr = f"output_oracle_gev.wav"
    file_ref_tgt = "test_target_ref.wav"
    file_ref_int = "test_interferer_ref.wav"
    
    if not os.path.exists(file_maxsnr):
        print(f"Error: '{file_maxsnr}' not found. Run test_maxsnr.py first.")
        return
    if not os.path.exists(file_ref_tgt):
        print(f"Error: '{file_ref_tgt}' not found. Run test_world.py first.")
        return

    # 2. Load
    s_est, _ = sf.read(file_maxsnr, dtype='float32')
    s_tgt, _ = sf.read(file_ref_tgt, dtype='float32')
    s_int, _ = sf.read(file_ref_int, dtype='float32')

    # 3. Calculate
    sdr_m, sir_m = calculate_metrics_manual(s_est, s_tgt, s_int)

    # 4. Report
    print("\n=== RESULTS ===")
    print(f"Max-SNR GEV (FOV={fov_width_for_test} deg):")
    print(f"  SIR: {sir_m:.2f} dB")
    print(f"  SDR: {sdr_m:.2f} dB")
    print("="*20)

if __name__ == "__main__":
    main()