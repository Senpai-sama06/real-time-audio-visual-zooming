import world
import debug_src
import masked_mvdr
import validate 

def main():
    print(">>> PIPELINE START <<<")

    # --- STAGE 1: World Generation ---
    output_dir_world = world.main()

    if output_dir_world:
        print(f"\n>>> Audio Generation Complete. Path: {output_dir_world}")
        
        # --- STAGE 2: Debug Analysis ---
        print("\n>>> STARTING DEBUGGER...")
        debug_src.main(output_dir_world)
        
        # --- STAGE 3: MVDR Processing ---
        print("\n>>> STARTING MVDR BEAMFORMER...")
        masked_mvdr.main(output_dir_world)

        # --- STAGE 4: Validation ---
        print("\n>>> STARTING VALIDATION METRICS...")
        validate.main(output_dir_world)
        
        print("\n>>> PIPELINE SUCCESS <<<")
        
    else:
        print("\n>>> PIPELINE FAILED (World generation returned None) <<<")

if __name__ == "__main__":
    main()