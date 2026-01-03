import numpy as np
from config import SystemConfig
from classifier import SpatioSpectralClassifier
from anchors import PhysicsAnchors

def test_classifier_physics():
    print("="*60)
    print("      CLASSIFIER INTERNAL STATE DIAGNOSTIC (WARMUP TEST)")
    print("="*60)

    # 1. SETUP
    TARGET_ANGLE = 1.5708 # 90 degrees
    MIC_DIST = 0.08       # 8 cm
    cfg = SystemConfig(mic_dist=MIC_DIST, target_angle=TARGET_ANGLE)
    classifier = SpatioSpectralClassifier(target_angle_rad=cfg.target_angle)
    
    # 2. PROBE 2: Synthetic Signal Injection (Time Series)
    print(f"\n[PROBE 2] Synthetic Signal Injection (50 Frames)")
    print("  > Injecting [1.0, 1.0] (Perfect Broadside)...")
    
    Y_frame_perfect = np.ones((cfg.n_fft // 2 + 1, 2), dtype=complex)
    Y_frame_perfect += np.random.randn(*Y_frame_perfect.shape) * 1e-6
    
    bin_idx = 50
    states_log = []
    
    # Run 50 frames to clear warmup
    for i in range(50):
        decision = classifier.process_frame(Y_frame_perfect)
        states_log.append(decision[bin_idx])
        
    print(f"  > First 5 Decisions: {states_log[:5]}")
    print(f"  > Final 5 Decisions: {states_log[-5:]}")
    
    final_state = states_log[-1]
    
    if final_state == 1:
        print(f"  [PASS] System Woke Up! State {final_state} (Target) achieved after warmup.")
    elif final_state == 5:
        print(f"  [FAIL] System Awake but Confused. State {final_state} (Jammer). Check Phase logic.")
    elif final_state == 0:
        print(f"  [FAIL] System Comatose. Still State 0 (Silence) after 50 frames.")
        print("         - Check if noise_floor is initializing to 1.0 instead of 1e-6.")

        
if __name__ == "__main__":
    test_classifier_physics()