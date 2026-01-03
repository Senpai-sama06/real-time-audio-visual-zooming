import numpy as np
import sys
from scipy.signal import stft

# Import the System
from classifier import SpatioSpectralClassifier
from config import SystemConfig

# ==========================================
# TEST HARNESS UTILITIES
# ==========================================
class SanityTester:
    def __init__(self):
        self.cfg = SystemConfig()
        self.fs = self.cfg.fs
        self.n_fft = self.cfg.n_fft
        self.classifier = SpatioSpectralClassifier(target_angle_rad=1.57)
        self.results_log = []

    def _generate_stft(self, audio_2ch):
        """Converts time-domain audio to STFT format used by classifier."""
        _, _, Zxx = stft(audio_2ch[:,0], fs=self.fs, nperseg=self.n_fft)
        _, _, Zyy = stft(audio_2ch[:,1], fs=self.fs, nperseg=self.n_fft)
        # Stack: [Bins, Frames, 2]
        return np.stack([Zxx, Zyy], axis=-1)

    def run_test(self, name, description, audio, expected_hypo, min_accuracy=0.95, check_freq_range=None, warmup=10):
        """
        Executes a test case.
        warmup: Number of initial frames to discard (allows filters to settle).
        """
        print(f"Running {name}...", end=" ", flush=True)
        
        # 1. Reset State (Important for independent tests)
        self.classifier = SpatioSpectralClassifier(target_angle_rad=1.57)
        
        # 2. Process
        Y_stft = self._generate_stft(audio)
        n_bins, n_frames, _ = Y_stft.shape
        decisions = np.zeros((n_bins, n_frames), dtype=int)
        
        for i in range(n_frames):
            decisions[:, i] = self.classifier.process_frame(Y_stft[:, i, :])

        # 3. Validate
        # Determine which bins to check
        start_bin = 0
        end_bin = n_bins
        if check_freq_range:
            start_bin, end_bin = check_freq_range

        # Extract relevant region AND discard warmup frames
        valid_frames_start = min(warmup, n_frames-1)
        roi = decisions[start_bin:end_bin, valid_frames_start:]
        
        # Calculate Accuracy
        total_pixels = roi.size
        
        if expected_hypo == 7:
            # For Aliasing test, we specifically look for 7
            matches = np.sum(roi == 7)
        else:
            # For other tests, we ignore bins that naturally fall into aliasing/DC
            # Valid mask: bins that are NOT H7 and NOT H0 (unless expecting H0)
            valid_mask = (roi != 7) 
            if expected_hypo != 0:
                valid_mask &= (roi != 0) # Ignore silence if checking for signal type
            
            if np.sum(valid_mask) == 0:
                accuracy = 0.0 
            else:
                matches = np.sum(roi[valid_mask] == expected_hypo)
                total_pixels = np.sum(valid_mask)
        
        accuracy = matches / (total_pixels + 1e-9)
        
        # 4. Log
        status = "PASS" if accuracy >= min_accuracy else "FAIL"
        color = "\033[92m" if status == "PASS" else "\033[91m"
        reset = "\033[0m"
        
        print(f"{color}{status}{reset} ({accuracy*100:.1f}%)")
        
        self.results_log.append({
            "name": name,
            "desc": description,
            "expect": f"H{expected_hypo}",
            "acc": f"{accuracy*100:.1f}%",
            "status": status
        })

    def print_scorecard(self):
        print("\n" + "="*80)
        print(f"{'SANITY CHECK SCORECARD':^80}")
        print("="*80)
        print(f"{'TEST NAME':<20} | {'EXPECTED':<10} | {'ACCURACY':<10} | {'STATUS':<10}")
        print("-" * 60)
        for r in self.results_log:
            color = "\033[92m" if r["status"] == "PASS" else "\033[91m"
            reset = "\033[0m"
            print(f"{r['name']:<20} | {r['expect']:<10} | {r['acc']:<10} | {color}{r['status']}{reset}")
        print("="*80 + "\n")

# ==========================================
# SIGNAL GENERATORS
# ==========================================
def gen_silence(fs, dur=1.0):
    """Absolute zeros."""
    return np.zeros((int(fs*dur), 2))

def gen_mono_target(fs, dur=1.0):
    """Perfectly correlated Broadside source."""
    t = np.linspace(0, dur, int(fs*dur))
    sig = np.random.randn(len(t)) * 0.1 # -20dBFS
    return np.vstack([sig, sig]).T

def gen_diffuse(fs, dur=1.0):
    """Uncorrelated White Noise."""
    sig = np.random.randn(int(fs*dur), 2) * 0.1
    return sig

def gen_aliasing_sweep(fs, dur=1.0):
    """Sine sweep going past the aliasing frequency."""
    from scipy.signal import chirp
    t = np.linspace(0, dur, int(fs*dur))
    # Sweep 0 to Nyquist
    sig = chirp(t, f0=100, f1=fs/2, t1=dur, method='linear') * 0.1
    return np.vstack([sig, sig]).T

def gen_gain_test(fs, db_level):
    """Mono target at specific dB level."""
    t = np.linspace(0, 1.0, int(fs*1.0))
    sig = np.random.randn(len(t))
    scale = 10**(db_level/20)
    sig = sig * scale
    return np.vstack([sig, sig]).T

def gen_radar_sweep(fs, mic_dist, c, dur=2.0):
    """Source moving from -90 to +90 degrees."""
    # We approximate with static chunks for simplicity
    chunks = []
    angles = [-90, -45, 0, 45, 90]
    samples_per_chunk = int(fs * (dur / len(angles)))
    
    for ang in angles:
        # Create steering vector delay
        tau = mic_dist * np.cos(np.deg2rad(ang)) / c
        delay_samples = int(tau * fs)
        
        noise = np.random.randn(samples_per_chunk) * 0.1
        ch1 = noise
        ch2 = np.roll(noise, delay_samples)
        
        chunks.append(np.vstack([ch1, ch2]).T)
        
    return np.concatenate(chunks, axis=0)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    tester = SanityTester()
    fs = tester.fs
    
    print("Initializing Sanity Protocol...")
    print(f"Sampling Rate: {fs}Hz | FFT: {tester.n_fft}")
    print("-" * 40)

    # --- TEST 1: ABSOLUTE NULL ---
    tester.run_test(
        "Null Check", 
        "Digital Zeros", 
        gen_silence(fs), 
        expected_hypo=0, # H0
        min_accuracy=0.99
    )

    # --- TEST 2: MONO TARGET ---
    tester.run_test(
        "Mono Check", 
        "Perfect Correlation (0 deg)", 
        gen_mono_target(fs), 
        expected_hypo=1 # H1
    )

    # --- TEST 3: DIFFUSE NOISE ---
    tester.run_test(
        "Diffuse Check", 
        "Uncorrelated Noise", 
        gen_diffuse(fs), 
        expected_hypo=6 # H6
    )

    # --- TEST 4: GAIN LINEARITY (LOUD) ---
    tester.run_test(
        "Gain High (-6dB)", 
        "Loud Mono Source", 
        gen_gain_test(fs, -6), 
        expected_hypo=1
    )

    # --- TEST 5: GAIN LINEARITY (QUIET) ---
    tester.run_test(
        "Gain Low (-50dB)", 
        "Quiet Mono Source", 
        gen_gain_test(fs, -50), 
        expected_hypo=1 # Should be H1 if noise floor init is correct
    )

    # --- TEST 6: PHYSICS/ALIASING ---
    alias_freq = 343.0 / (2 * 0.05) # ~3430 Hz
    alias_bin = int(alias_freq / (fs/tester.n_fft))
    
    tester.run_test(
        "Physics Check", 
        "Frequency > Aliasing Limit", 
        gen_aliasing_sweep(fs), 
        expected_hypo=7, # H7
        check_freq_range=(alias_bin + 5, tester.n_fft // 2),
        min_accuracy=0.95 # Relaxed
    )

    # --- TEST 7: RADAR SWEEP (Broadside vs Interferer) ---
    radar_sig = gen_radar_sweep(fs, 0.05, 343.0)
    # The middle 20% of this signal is at 0 degrees.
    mid_start = int(len(radar_sig) * 0.4)
    mid_end = int(len(radar_sig) * 0.6)
    
    tester.run_test(
        "Radar (Broadside)", 
        "Source passing 0 deg", 
        radar_sig[mid_start:mid_end], 
        expected_hypo=1
    )

    # The start (90 deg) should be H5
    tester.run_test(
        "Radar (Endfire)", 
        "Source at 90 deg", 
        radar_sig[0:int(len(radar_sig)*0.2)], 
        expected_hypo=5 # H5 Interferer
    )

    # --- FINAL REPORT ---
    tester.print_scorecard()