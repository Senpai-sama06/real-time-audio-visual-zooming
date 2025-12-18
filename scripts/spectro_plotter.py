import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os # Import os for file existence check

def plot_three_spectrograms_comparison(filepath_original, filepath_filtered, filepath_target):
    """
    Reads three separate audio files and plots their spectrograms 
    side by side for visual comparison: Original, Filtered, and Target.

    Args:
        filepath_original (str): Path to the original audio signal.
        filepath_filtered (str): Path to the processed/filtered audio signal.
        filepath_target (str): Path to the reference/target audio signal.
    """
    
    file_paths = {
        "Original": filepath_original,
        "Filtered": filepath_filtered,
        "Target": filepath_target
    }
    
    loaded_data = {}
    
    # 1. --- Read and Process All Three Files ---
    rate = None
    for name, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Error: {name} file not found at {path}")
            return

        try:
            current_rate, current_data = wavfile.read(path)
            
            # Use the rate of the first successfully loaded file as reference
            if rate is None:
                rate = current_rate
            elif current_rate != rate:
                print(f"Warning: {name} file has a different sampling rate ({current_rate} Hz) than the first ({rate} Hz).")
            
            # Convert to mono if necessary and convert to float
            if current_data.ndim > 1:
                current_data = current_data[:, 0]
            
            loaded_data[name] = current_data.astype(np.float64)
            
        except Exception as e:
            print(f"An error occurred while reading {name} file ({path}): {e}")
            return

    if rate is None:
        print("Could not load any audio files.")
        return

    # 2. --- Setup the Figure for Three Side-by-Side Plots ---
    # 1 row, 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('Three-Way Spectrogram Comparison: Original, Filtered, and Target Signal', fontsize=16)

    # Common Spectrogram Parameters
    NFFT = 256
    N_OVERLAP = 128
    
    # --- Plot 1: Original Signal ---
    Pxx1, freqs1, bins1, im1 = ax1.specgram(
        loaded_data["Original"], 
        NFFT=NFFT, 
        Fs=rate, 
        noverlap=N_OVERLAP, 
        cmap='viridis'
    )
    ax1.set_title("1. Original Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")

    # --- Plot 2: Filtered Signal ---
    Pxx2, freqs2, bins2, im2 = ax2.specgram(
        loaded_data["Filtered"], 
        NFFT=NFFT, 
        Fs=rate, 
        noverlap=N_OVERLAP, 
        cmap='viridis'
    )
    ax2.set_title("2. Filtered Signal")
    ax2.set_xlabel("Time (s)")

    # --- Plot 3: Target Signal ---
    # We use the image object from the third plot for the color bar
    Pxx3, freqs3, bins3, im3 = ax3.specgram(
        loaded_data["Target"], 
        NFFT=NFFT, 
        Fs=rate, 
        noverlap=N_OVERLAP 
        # cmap='viridis'
    )
    ax3.set_title("3. Target/Reference Signal")
    ax3.set_xlabel("Time (s)")

    # 3. --- Add a Single Colorbar ---
    # Create an axis for the colorbar
    # Coordinates: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7]) 
    fig.colorbar(im3, cax=cbar_ax, label='Intensity (dB)')

    # 4. --- Adjust layout and Display ---
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust for the colorbar
    plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    # NOTE: Replace these with the actual paths to your three audio files.
    file_path_original = '/home/rpzrm/global/projects/real-time-audio-visual-zooming/Final_pipeline/data/simulated/batch_test_000/mixture.wav' 
    file_path_filtered = '/home/rpzrm/global/projects/real-time-audio-visual-zooming/Final_pipeline/data/results/batch_test_000_results/batch_test_000_enhanced.wav' 
    file_path_target = '/home/rpzrm/global/projects/real-time-audio-visual-zooming/Final_pipeline/data/simulated/batch_test_000/target.wav' 
    
    # Run the function
    plot_three_spectrograms_comparison(
        file_path_original, 
        file_path_filtered, 
        file_path_target
    )