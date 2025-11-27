import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from pesq import pesq
from typing import Union, Tuple
import os

# --- NEW: Import Core Sampling Rate Constant ---
# Ensure the PESQ evaluation uses the exact FS defined in the core simulation.
from rt_av_zoom.core.world import FS 

class PESQEvaluator:
    """
    An object-oriented class for calculating PESQ and plotting spectrograms
    between a reference (source) and a degraded (reconstructed) audio file.
    """
    # Use the imported FS as a default if none is provided or inferred.
    def __init__(self, ref_file_path: str, deg_file_path: str, target_fs: int = FS):
        """
        Initializes the evaluator with file paths and loads the audio data.
        """
        self.ref_file_path = ref_file_path
        self.deg_file_path = deg_file_path
        self.fs: Union[int, None] = target_fs # Use imported FS as target
        self.ref_audio: Union[np.ndarray, None] = None
        self.deg_audio: Union[np.ndarray, None] = None
        
        self._load_audio()

    def _load_audio(self):
        """
        Loads the reference and degraded audio files using scipy.io.wavfile.
        """
        try:
            # Read reference audio
            ref_fs, self.ref_audio = wavfile.read(self.ref_file_path)
            
            # Read degraded audio
            deg_fs, self.deg_audio = wavfile.read(self.deg_file_path)

        except FileNotFoundError as e:
            print(f"Error: File not found: {e}")
            raise
        except Exception as e:
            print(f"Error reading WAV files: {e}")
            raise

        # Check sampling rate consistency
        if ref_fs != deg_fs or ref_fs != self.fs:
             print(f"Warning: Audio FS mismatch (Reference: {ref_fs}Hz, Target: {self.fs}Hz).")
             # We rely on the core FS, but use the file's FS if it's consistent.
             if ref_fs == deg_fs: self.fs = ref_fs
             else: raise ValueError("Audio files have different sampling rates.")

        # Check for mono and convert to float 
        if self.ref_audio.ndim > 1 or self.deg_audio.ndim > 1:
             raise ValueError("PESQ requires mono audio. Please convert multi-channel files to mono.")
        
        # Ensure audio arrays have the same length (a requirement for PESQ)
        min_len = min(len(self.ref_audio), len(self.deg_audio))
        self.ref_audio = self.ref_audio[:min_len]
        self.deg_audio = self.deg_audio[:min_len]
        
        self.ref_audio = self.ref_audio.astype(np.float64)
        self.deg_audio = self.deg_audio.astype(np.float64)
        
        print(f"Audio loaded. Sampling Rate (fs): {self.fs} Hz, Length: {min_len} samples.")


    def calculate_pesq(self, mode: str) -> float:
        """
        Calculates the PESQ score for a given mode.
        """
        if self.fs is None or self.ref_audio is None or self.deg_audio is None:
            raise RuntimeError("Audio data was not loaded correctly.")

        # Allow 16000 Hz for Narrow-Band, as the pesq library handles downsampling.
        if mode == 'nb' and self.fs not in [8000, 16000]:
            raise ValueError(
                f"Narrow-Band PESQ requires 8000 Hz or 16000 Hz input, but audio is {self.fs} Hz."
            )
        # Wide-Band requires exactly 16000 Hz.
        elif mode == 'wb' and self.fs != 16000:
            raise ValueError(
                f"Wide-Band PESQ requires 16000 Hz, but audio is {self.fs} Hz."
            )
        elif mode not in ['nb', 'wb']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'nb' or 'wb'.")

        score = pesq(self.fs, self.ref_audio, self.deg_audio, mode)
        return score

    def evaluate(self) -> Tuple[float, float]:
        """
        Runs both Narrow-Band and Wide-Band evaluation, if supported by the fs.
        """
        nb_score = 0.0
        wb_score = 0.0

        if self.fs == 16000:
            nb_score = self.calculate_pesq('nb')
            wb_score = self.calculate_pesq('wb')
            print(f"Narrow-Band (NB) PESQ: {nb_score:.4f}")
            print(f"Wide-Band (WB) PESQ: {wb_score:.4f}")
        elif self.fs == 8000:
            nb_score = self.calculate_pesq('nb')
            print(f"Narrow-Band (NB) PESQ: {nb_score:.4f}")
            print(f"Wide-Band (WB) PESQ: 0.0000 (Unsupported)")
        else:
            print(f"PESQ is only defined for 8000 Hz (NB) or 16000 Hz (NB/WB). Current fs is {self.fs} Hz. Skipping calculation.")

        return nb_score, wb_score

    # ... (plot_spectograms function body remains the same) ...

    def plot_spectograms(self, window_size: float = 0.02, overlap_ratio: float = 0.5):
        """
        Calculates and plots the spectrograms for the reference and degraded audio.
        """
        if self.fs is None or self.ref_audio is None or self.deg_audio is None:
            print("Cannot plot: Audio data was not loaded correctly.")
            return

        # Convert window size in seconds to number of samples (NPERSEG)
        NPERSEG = int(self.fs * window_size)
        NOVERLAP = int(NPERSEG * overlap_ratio)
        
        # 1. Calculate spectrograms
        # Target/Reference Audio
        f_ref, t_ref, Sxx_ref = spectrogram(self.ref_audio, self.fs, nperseg=NPERSEG, noverlap=NOVERLAP)
        # Degraded Audio
        f_deg, t_deg, Sxx_deg = spectrogram(self.deg_audio, self.fs, nperseg=NPERSEG, noverlap=NOVERLAP)

        # 2. Setup the plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'Spectrogram Comparison (Sampling Rate: {self.fs} Hz)', fontsize=14)

        # Helper function to plot a single spectrogram
        def plot_single_spec(ax, t, f, Sxx, title):
            # Convert power spectrogram to dB
            Sxx_db = 10 * np.log10(Sxx + 1e-6) # Add small epsilon to avoid log(0)
            
            im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
            ax.set_title(title, fontsize=12)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_ylim([0, self.fs / 2]) # Limit frequency to Nyquist
            fig.colorbar(im, ax=ax, format='%+2.0f dB')

        # 3. Plot Reference Spectrogram
        plot_single_spec(axes[0], t_ref, f_ref, Sxx_ref, f'Reference Audio: {self.ref_file_path}')

        # 4. Plot Degraded Spectrogram
        plot_single_spec(axes[1], t_deg, f_deg, Sxx_deg, f'Degraded Audio: {self.deg_file_path}')
        axes[1].set_xlabel('Time (s)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# --- Example Usage ---
# NOTE: Replace these with your actual file paths.
REF_AUDIO_PATH = "target_reference.wav"
DEG_AUDIO_PATH = "output_oracle.wav"

if __name__ == '__main__':
    try:
        # Create an instance of the PESQEvaluator object, uses the imported FS
        evaluator = PESQEvaluator(REF_AUDIO_PATH, DEG_AUDIO_PATH)
        
        # Run the full evaluation (PESQ Scores)
        nb_score, wb_score = evaluator.evaluate()
        
        # Display results in a final summary
        print("\n--- PESQ Summary ---")
        if nb_score > 0.0:
            print(f"Narrow-Band PESQ Score: {nb_score:.4f}")
        if wb_score > 0.0:
            print(f"Wide-Band PESQ Score: {wb_score:.4f}")
            
        # Plot the Spectrograms
        print("\n--- Visualizing Spectrograms ---")
        evaluator.plot_spectograms(window_size=0.032) # Using a 32ms window
        
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")