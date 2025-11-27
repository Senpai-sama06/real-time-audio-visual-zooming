import os
import sys
import subprocess
# Import core modules from the installed package: rt_av_zoom.core
from rt_av_zoom.core import world
from rt_av_zoom.core import masked_mvdr 

def main():
    print(">>> PIPELINE START <<<")

    # --- STAGE 1: World Generation (Core Package) ---
    # This generates the multi-channel mixture and saves references.
    output_dir_world = world.main() 

    if output_dir_world:
        # output_dir_world will look like: Simulation_Output_20251127_.../World_Outputs
        print(f"\n>>> Audio Generation Complete. Path: {output_dir_world}")
        
        # --- STAGE 2: Debug Analysis (Scripts Sub-process) ---
        # We execute the script debug_srp.py, passing the output path as an argument.
        print("\n>>> STARTING DEBUGGER (SRP Scan)...")
        try:
            subprocess.run(
                ["python", os.path.join("scripts", "debug_srp.py"), output_dir_world], 
                check=True # check=True raises an error if the script fails
            )
        except subprocess.CalledProcessError:
            print("ERROR: Debug script failed. Check debug_srp.py logs.")
            return

        # --- STAGE 3: MVDR Processing (Core Package) ---
        # This enhances the audio and saves the output WAV file.
        print("\n>>> STARTING MVDR BEAMFORMER...")
        masked_mvdr.main(output_dir_world)

        # --- STAGE 4: Validation (Scripts Sub-process) ---
        # We execute the script run_metrics.py, passing the output path as an argument.
        print("\n>>> STARTING VALIDATION METRICS...")
        try:
            subprocess.run(
                ["python", os.path.join("scripts", "run_metrics.py"), output_dir_world], 
                check=True
            )
        except subprocess.CalledProcessError:
            print("ERROR: Validation script failed. Check run_metrics.py logs.")
            return

        print("\n>>> PIPELINE SUCCESS <<<")
        
    else:
        print("\n>>> PIPELINE FAILED (World generation returned None) <<<")

if __name__ == "__main__":
    # Ensure the script can be run from the project root
    main()