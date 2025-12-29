import argparse
import time
import os
from src import simulation, inference, metrics, config

# Optional: Progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

def run_batch(n_runs, start_idx=0, n_interferers=2):
    model_path = os.path.join("models", "mask_estimator_phy.tflite")
    
    print(f"=== STARTING BATCH RUN: {n_runs} Iterations ===")
    
    # Loop
    for i in tqdm(range(start_idx, start_idx + n_runs), desc="Processing"):
        run_name = f"batch_test_{i:03d}" # e.g., batch_test_001
        
        try:
            # 1. Simulate
            mix_path = simulation.generate_scene(
                run_name=run_name,
                dataset='ljspeech', # or 'musan'
                reverb=True,
                n_interferers=n_interferers,
                snr_target=50
            )
            
            if not mix_path: continue # Skip if sim failed

            # 2. Inference
            inference.enhance_audio(
                run_name=run_name,
                input_path=mix_path,
                model_path=model_path
            )

            # 3. Evaluate (This now auto-logs to CSV)
            metrics.evaluate_run(run_name)
            
            # Optional: Clean up huge wav files to save disk space
            # import shutil
            # shutil.rmtree(os.path.join(config.SIM_DIR, run_name))

        except Exception as e:
            print(f"\n[ERROR] Failed on {run_name}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help="Number of runs")
    parser.add_argument('--start', type=int, default=0, help="Start index for naming")
    parser.add_argument('--interferers', type=int, default=2, help="Number of interferers")
    
    args = parser.parse_args()
    
    run_batch(args.n, args.start, args.interferers)