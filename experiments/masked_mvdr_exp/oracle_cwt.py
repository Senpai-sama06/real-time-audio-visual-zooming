#!/usr/bin/env python3
"""
oracle_wavelet_mvdr.py

Wavelet-domain oracle MVDR audio-zoom pipeline (complete script).

Usage: edit INPUT_* paths below or run as-is if your world builder produced the expected files.

Author: ChatGPT (GPT-5 Thinking mini)
"""

import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pywt  # Using PyWavelets
from scipy.ndimage import gaussian_filter
import warnings
from tqdm import tqdm  # <--- ADDED

# -------------------------
# USER PARAMETERS (edit)
# -------------------------
# Input WAVs (mixture should have shape (n_mics, n_samples) stored as multi-channel wav or separate single-channel files)
INPUT_MIXTURE_WAV = "samples/mixture_3_sources.wav"           # multi-channel wav (2 channels expected)
INPUT_TARGET_REF_WAV = "samples/target_reference.wav"   # single-channel -- oracle target-only reference
INPUT_INTERF_REF_WAV = "samples/interference_reference.wav"  # single-channel -- oracle interference-only reference

# If you want the script to try to call your world builder to produce the above files, set this True.
TRY_IMPORT_WORLD_PY = False
WORLD_PY_PATH = "/mnt/data/world.py"  # path from your uploaded files

OUTPUT_ZOOM_WAV = "samples/oracle_op_cwt.wav"

# Wavelet / analysis parameters
FS = None          # If None, read from files; otherwise override sample rate
NUM_SCALES = 256
F_MIN = 50.0       # min center frequency (Hz)
F_MAX = 8000.0     # max center frequency (Hz)
MORLET_W = 6.0     # Morlet parameter (bandwidth parameter in cmor)

# Beamforming / geometry
CENTER_ANGLE_DEG = 90.0   # broadside angle in degrees (your target)
ZOOM_DEG = 30.0           # half-width of angular acceptance in degrees (e.g. 30 -> 90 +/- 30)
MIC_SPACING = 0.08        # meters (distance between the two mics; if they are at ±d/2 model)
SPEED_OF_SOUND = 343.0    # m/s

# Oracle mask parameters
RATIO_THRESHOLD = 1.2     # use mag_tgt / (mag_int + eps) >= RATIO_THRESHOLD for oracle
SMOOTH_MASK_SIGMA = 1.0   # gaussian smoothing sigma (scales × time)
POWER_GATE_MIN = 1e-4     # minimal normalized combined power to consider active

# MVDR parameters
DIAGONAL_LOADING = 1e-5   # absolute loading added to diagonal of Rn
MIN_NOISE_FRAMES = 3      # if too few noise frames for a scale, fall back to global loading

# Diagnostics / plotting
PLOT_DIAGNOSTICS = True
SAVE_PLOTS_TO = "/mnt/data/wavelet_diagnostics.png"

# -------------------------
# Helper utilities
# -------------------------
def try_import_world_and_generate(world_path, out_mixture, out_tgt, out_int):
    """
    Best-effort: try to import the uploaded world.py and invoke a function to produce mixture/ref files.
    """
    if not os.path.exists(world_path):
        return False
    try:
        # Add containing dir to sys.path and import
        world_dir = os.path.dirname(world_path) or "."
        if world_dir not in sys.path:
            sys.path.insert(0, world_dir)
        import importlib, importlib.util
        spec = importlib.util.spec_from_file_location("user_world_module", world_path)
        world = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(world)
        # Try several plausible functions
        for fn in ("generate_world", "build_world", "main", "run", "create_world"):
            if hasattr(world, fn):
                try:
                    tqdm.write(f"[world.py] calling function '{fn}()' to generate files.")
                    getattr(world, fn)()
                except TypeError:
                    try:
                        getattr(world, fn)(out_mixture, out_tgt, out_int)
                    except Exception as e:
                        tqdm.write("[world.py] attempted to call with and without args; continuing.")
                return True
        return False
    except Exception as e:
        tqdm.write(f"[world.py] import/execute error: {e}")
        return False

def read_wav_multichannel(path):
    """Read wav; returns (data, fs)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data, fs = sf.read(path)
    return data, fs

# -------------------------
# Wavelet transform helpers
# -------------------------
def cwt_complex(x, fs, num_scales=NUM_SCALES, f_min=F_MIN, f_max=F_MAX, w=MORLET_W):
    """
    Compute complex CWT using PyWavelets.
    """
    freqs = np.geomspace(f_min, f_max, num_scales)
    wavelet_name = f"cmor{w}-1.0"
    dt = 1.0 / fs
    scales = 1.0 / (freqs * dt)
    
    # pywt.cwt returns (coefficients, frequencies)
    W, _ = pywt.cwt(x, scales, wavelet_name, sampling_period=dt)
    return W, freqs, scales

def wrap_angle(a):
    """Wrap to [-pi, pi]"""
    return (a + np.pi) % (2*np.pi) - np.pi

# -------------------------
# Main pipeline
# -------------------------
def main():
    global FS

    # Initialize overall pipeline progress bar
    # Steps: 1.Inputs 2.ReadWav 3.CWT 4.Oracle 5.Geo 6.MVDR 7.iCWT 8.Write 9.Plot
    total_steps = 9
    with tqdm(total=total_steps, desc="Pipeline Progress", unit="step") as main_pbar:

        # 1) Ensure input WAVs exist or try world.py
        main_pbar.set_description("Step 1/9: Checking inputs")
        missing = []
        for p in (INPUT_MIXTURE_WAV, INPUT_TARGET_REF_WAV, INPUT_INTERF_REF_WAV):
            if not os.path.exists(p):
                missing.append(p)
        if missing and TRY_IMPORT_WORLD_PY:
            tqdm.write("[info] Missing input files, running world.py...")
            success = try_import_world_and_generate(WORLD_PY_PATH, INPUT_MIXTURE_WAV, INPUT_TARGET_REF_WAV, INPUT_INTERF_REF_WAV)
            if not success:
                tqdm.write("[warn] world.py generation not successful.")
        
        missing = [p for p in (INPUT_MIXTURE_WAV, INPUT_TARGET_REF_WAV, INPUT_INTERF_REF_WAV) if not os.path.exists(p)]
        if missing:
            tqdm.write("[error] Missing required input files.")
            raise FileNotFoundError("Required input WAVs not found.")
        main_pbar.update(1)

        # 2) Read mixture
        main_pbar.set_description("Step 2/9: Reading WAVs")
        mix_data, fs = read_wav_multichannel(INPUT_MIXTURE_WAV)
        if FS is None:
            FS = fs
        elif FS != fs:
            warnings.warn(f"Overriding sample rate with {fs}.")
            FS = fs

        if mix_data.ndim == 1:
            raise ValueError("Mixture file appears mono.")
        elif mix_data.ndim == 2:
            samples, nch = mix_data.shape
            if nch < 2:
                raise ValueError("Mixture must have at least 2 channels.")
            y_mix = mix_data[:, :2].T
        else:
            raise ValueError("Unexpected mixture WAV shape.")

        tgt_ref, fs_t = read_wav_multichannel(INPUT_TARGET_REF_WAV)
        int_ref, fs_i = read_wav_multichannel(INPUT_INTERF_REF_WAV)
        if tgt_ref.ndim > 1: tgt_ref = tgt_ref[:, 0]
        if int_ref.ndim > 1: int_ref = int_ref[:, 0]

        if fs_t != FS or fs_i != FS:
            raise ValueError("All files must have same sample rate.")

        global_scale = max(np.max(np.abs(y_mix)), np.max(np.abs(tgt_ref)), np.max(np.abs(int_ref)), 1e-9)
        y_mix = y_mix / global_scale
        tgt_ref = tgt_ref / global_scale
        int_ref = int_ref / global_scale

        n_mics, n_samples = y_mix.shape
        tqdm.write(f"[info] fs={FS}, n_mics={n_mics}, n_samples={n_samples}")
        main_pbar.update(1)

        # 3) CWT
        main_pbar.set_description("Step 3/9: Computing CWT")
        tqdm.write("[info] computing CWT for mixture and references...")
        W1, freqs, widths = cwt_complex(y_mix[0], FS, num_scales=NUM_SCALES, f_min=F_MIN, f_max=F_MAX, w=MORLET_W)
        W2, _, _ = cwt_complex(y_mix[1], FS, num_scales=NUM_SCALES, f_min=F_MIN, f_max=F_MAX, w=MORLET_W)
        Wtgt, _, _ = cwt_complex(tgt_ref, FS, num_scales=NUM_SCALES, f_min=F_MIN, f_max=F_MAX, w=MORLET_W)
        Wint, _, _ = cwt_complex(int_ref, FS, num_scales=NUM_SCALES, f_min=F_MIN, f_max=F_MAX, w=MORLET_W)
        main_pbar.update(1)

        # 4) Oracle mask
        main_pbar.set_description("Step 4/9: Oracle Mask")
        tqdm.write("[info] computing oracle mask...")
        mag_tgt = np.abs(Wtgt)
        mag_int = np.abs(Wint)
        ratio = mag_tgt / (mag_int + 1e-12)
        M_oracle = (ratio >= RATIO_THRESHOLD).astype(float)
        main_pbar.update(1)

        # 5) Geometric mask
        main_pbar.set_description("Step 5/9: Geometric Mask")
        tqdm.write("[info] computing geometric mask...")
        delta_phi = np.angle(W1 * np.conj(W2))

        def expected_phase_for_angle(freqs, angle_deg, d, c):
            theta = np.deg2rad(angle_deg)
            tau = (d / 2.0) * np.cos(theta) / c
            phi = 2.0 * np.pi * freqs * tau
            return phi

        phi0 = expected_phase_for_angle(freqs, CENTER_ANGLE_DEG, MIC_SPACING, SPEED_OF_SOUND)
        phi_lo = expected_phase_for_angle(freqs, CENTER_ANGLE_DEG - ZOOM_DEG, MIC_SPACING, SPEED_OF_SOUND)
        phi_hi = expected_phase_for_angle(freqs, CENTER_ANGLE_DEG + ZOOM_DEG, MIC_SPACING, SPEED_OF_SOUND)
        eps_f = np.maximum(np.abs(phi_lo - phi0), np.abs(phi_hi - phi0)) + 1e-12

        phi0_mat = phi0[:, None]
        dev = np.abs(wrap_angle(delta_phi - phi0_mat))
        M_angle = (dev <= eps_f[:, None]).astype(float)

        power = (np.abs(W1)**2 + np.abs(W2)**2) / 2.0
        power_norm = power / (np.max(power) + 1e-12)
        M_power = (power_norm >= POWER_GATE_MIN).astype(float)

        M_final = M_oracle * M_angle * M_power
        if SMOOTH_MASK_SIGMA > 0:
            M_final = gaussian_filter(M_final, sigma=SMOOTH_MASK_SIGMA)
        main_pbar.update(1)

        # 6) Per-scale MVDR
        main_pbar.set_description("Step 6/9: MVDR Estimation")
        tqdm.write("[info] estimating MVDR weights per scale...")
        num_scales = W1.shape[0]
        W_out = np.zeros_like(W1, dtype=complex)

        def steering_vector(freq, angle_deg, d, c):
            theta = np.deg2rad(angle_deg)
            tau = (d / 2.0) * np.cos(theta) / c
            phi = 2.0 * np.pi * freq * tau
            return np.array([1.0, np.exp(-1j * phi)], dtype=complex)[:, None]

        global_noise_power = np.mean(power) + 1e-12
        
        # Using a nested tqdm bar for the loop over scales
        for s in tqdm(range(num_scales), desc="MVDR Scales", leave=False):
            noise_mask_cols = np.where((M_final[s, :] < 0.5))[0]
            Y = np.vstack([W1[s, :], W2[s, :]])
            if noise_mask_cols.size >= MIN_NOISE_FRAMES:
                Y_noise = Y[:, noise_mask_cols]
                Rn = (Y_noise @ Y_noise.conj().T) / max(1, Y_noise.shape[1])
            else:
                Rn = np.eye(2, dtype=complex) * (global_noise_power + 1e-9)

            Rn += DIAGONAL_LOADING * np.eye(2, dtype=complex)
            dvec = steering_vector(freqs[s], CENTER_ANGLE_DEG, MIC_SPACING, SPEED_OF_SOUND)
            try:
                w = np.linalg.solve(Rn, dvec)
                denom = (dvec.conj().T @ w)[0, 0]
                if np.abs(denom) < 1e-12:
                    w = w * 0.0
                else:
                    w = w / denom
            except np.linalg.LinAlgError:
                w = dvec / 2.0
            W_out[s, :] = (w.conj().T @ Y).ravel()
        main_pbar.update(1)

        # 7) Inverse CWT
        main_pbar.set_description("Step 7/9: Inverse CWT")
        tqdm.write("[info] reconstructing signal (iCWT)...")
        try:
            # PyWavelets icwt implementation
            wavelet_name = f"cmor{MORLET_W}-1.0"
            # Note: pywt.icwt requires PyWavelets >= 1.1
            y_out = pywt.icwt(W_out, widths, wavelet_name, sampling_period=1/FS)
        except Exception as e:
            tqdm.write(f"[error] iCWT failed: {e}")
            raise
        main_pbar.update(1)

        # 8) Write output
        main_pbar.set_description("Step 8/9: Saving WAV")
        y_out = y_out / (np.max(np.abs(y_out)) + 1e-12)
        sf.write(OUTPUT_ZOOM_WAV, y_out, FS)
        tqdm.write(f"[done] wrote output to: {OUTPUT_ZOOM_WAV}")
        main_pbar.update(1)

        # 9) Diagnostics
        main_pbar.set_description("Step 9/9: Plotting")
        if PLOT_DIAGNOSTICS:
            try:
                tqdm.write("[info] plotting diagnostics...")
                plt.figure(figsize=(14, 8))
                ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
                ax2 = plt.subplot2grid((3, 3), (0, 2))
                ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
                ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

                nT = W1.shape[1]
                ax1.imshow(M_final, origin='lower', aspect='auto', extent=[0, nT, freqs[0], freqs[-1]])
                ax1.set_title("Final mask (scale x time)")
                ax1.set_ylabel("freq (Hz)")
                ax1.set_xlabel("time frames")

                ax2.plot(freqs, eps_f)
                ax2.set_xscale('log')
                ax2.set_title("eps_f vs freq")

                med_dev = np.median(np.abs(wrap_angle(delta_phi - phi0_mat)), axis=1)
                ax3.plot(freqs, med_dev, label='median dev')
                ax3.plot(freqs, eps_f, label='eps_f')
                ax3.set_xscale('log')
                ax3.legend()

                t = np.arange(len(y_out)) / FS
                ax4.plot(t, y_out, label='zoom_out')
                ax4.plot(t[:len(y_mix[0])], y_mix[0] / (np.max(np.abs(y_mix[0])) + 1e-12), alpha=0.5, label='mic0')
                ax4.set_xlim(0, min(3.0, t[-1]))
                ax4.legend()

                plt.tight_layout()
                plt.savefig(SAVE_PLOTS_TO, dpi=200)
                tqdm.write(f"[info] saved diagnostics to: {SAVE_PLOTS_TO}")
                plt.close()
            except Exception as e:
                tqdm.write(f"[warn] failed to plot: {e}")
        main_pbar.update(1)
        main_pbar.set_description("Completed")

if __name__ == "__main__":
    main()