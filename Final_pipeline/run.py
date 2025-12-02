import argparse
import os
from src import simulation, inference, metrics, config

def main():
    parser = argparse.ArgumentParser(description="Neural Beamforming Pipeline")
    
    # Updated choices to include 'eval'
    parser.add_argument('mode', choices=['sim', 'inf', 'eval', 'full'], help="Action: sim, inf, eval, or full")
    
    parser.add_argument('--name', type=str, required=True, 
                        help="Run Name (e.g., 'test1'). Used to find folders.")

    # Sim Params
    parser.add_argument('--reverb', action='store_true', default=True)
    parser.add_argument('--no-reverb', action='store_false', dest='reverb')
    parser.add_argument('--dataset', default='ljspeech')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--snr', type=int, default=5)

    args = parser.parse_args()
    
    # Standardized Paths
    sim_folder = os.path.join(config.SIM_DIR, args.name)
    mixture_path = os.path.join(sim_folder, "mixture.wav")
    model_path = os.path.join(config.PROJECT_ROOT, "models", "mask_estimator_phy.tflite")

    # --- 1. SIMULATION ---
    if args.mode in ['sim', 'full']:
        print(f"\n--- 1. BUILDING WORLD: {args.name} ---")
        if os.path.exists(sim_folder) and args.mode == 'sim':
            print(f"Warning: {sim_folder} exists.")
        
        simulation.generate_scene(
            run_name=args.name,
            dataset=args.dataset,
            reverb=args.reverb,
            n_interferers=args.n,
            snr_target=args.snr
        )

    # --- 2. INFERENCE ---
    if args.mode in ['inf', 'full']:
        print(f"\n--- 2. RUNNING INFERENCE: {args.name} ---")
        if not os.path.exists(mixture_path):
            print(f"Error: Mixture not found. Run 'sim' first.")
            return

        inference.enhance_audio(
            run_name=args.name,
            input_path=mixture_path,
            model_path=model_path
        )

    # --- 3. EVALUATION ---
    # Runs if mode is 'eval', OR automatically after 'inf'/'full'
    if args.mode in ['eval', 'inf', 'full']:
        print(f"\n--- 3. EVALUATING RESULTS: {args.name} ---")
        metrics.evaluate_run(args.name)

if __name__ == "__main__":
    main()