import soundfile as sf
import os
import glob
import numpy as np

# Find latest run
base_dir = "simulation_results/ljspeech_anechoic_20251130_154029"
search_pattern = os.path.join(base_dir, '*_*_*')
all_runs = sorted(glob.glob(search_pattern), reverse=True)

if not all_runs:
    print("No simulation runs found.")
    exit()

target_dir = "simulation_results/ljspeech_anechoic_20251130_154029"
mix_path = os.path.join(target_dir, "mixture.wav")

print(f"Checking: {mix_path}")

try:
    data, fs = sf.read(mix_path)
    print(f"  Shape: {data.shape}")
    print(f"  Sample Rate: {fs}")
    print(f"  Data Type: {data.dtype}")
    
    # Check for silence or NaN
    if np.all(data == 0):
        print("  WARNING: File contains only zeros (silence).")
    if np.any(np.isnan(data)):
        print("  WARNING: File contains NaNs.")
        
    # Heuristic for shape correctness
    # We expect (Samples, Channels) from sf.read, e.g. (80000, 2)
    if data.shape[0] < data.shape[1]:
        print("  WARNING: Shape looks suspicious for sf.read. Expected (Samples, Channels).")
        print("           Did you save it transposed?")
    else:
        print("  Shape looks standard for soundfile read (Samples, Channels).")

except Exception as e:
    print(f"Error reading file: {e}")