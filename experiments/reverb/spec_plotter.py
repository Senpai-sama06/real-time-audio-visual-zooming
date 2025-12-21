import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrograms_side_by_side(audio_file_1, audio_file_2):
    """
    Plots the spectrograms of two audio files side by side.
    """
    
    # 1. Load the audio files
    # sr=None preserves the original sampling rate
    y1, sr1 = librosa.load(audio_file_1, sr=None)
    y2, sr2 = librosa.load(audio_file_2, sr=None)

    # 2. Compute the Short-Time Fourier Transform (STFT)
    # n_fft determines the window size, hop_length determines the overlap
    n_fft = 2048
    hop_length = 512
    
    D1 = librosa.stft(y1, n_fft=n_fft, hop_length=hop_length)
    D2 = librosa.stft(y2, n_fft=n_fft, hop_length=hop_length)

    # 3. Convert amplitude to decibels (dB) for visualization
    S_db1 = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    S_db2 = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

    # 4. Set up the plot (1 row, 2 columns)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # --- Plot First Audio ---
    img1 = librosa.display.specshow(S_db1, sr=sr1, hop_length=hop_length, 
                                    x_axis='time', y_axis='log', ax=axes[0], cmap='magma')
    axes[0].set_title(f'Spectrogram: {audio_file_1}')
    fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    # --- Plot Second Audio ---
    img2 = librosa.display.specshow(S_db2, sr=sr2, hop_length=hop_length, 
                                    x_axis='time', y_axis='log', ax=axes[1], cmap='magma')
    axes[1].set_title(f'Spectrogram: {audio_file_2}')
    fig.colorbar(img2, ax=axes[1], format="%+2.0f dB")

    # Final layout adjustments
    plt.tight_layout()
    plt.show()

# --- Usage ---
# Replace these strings with your actual file paths
file_path_1 = 'sample/output_oracle_mvdr_hrnr_gated.wav'
file_path_2 = 'sample/target.wav'

# Uncomment the line below to run it immediately if you have files ready
plot_spectrograms_side_by_side(file_path_1, file_path_2)    