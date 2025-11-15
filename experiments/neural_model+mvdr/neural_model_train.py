import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.signal
import os
import random
import glob
import librosa
import soundfile as sf
from tqdm import tqdm 
import warnings
import torch.nn.functional as F 
warnings.filterwarnings("ignore", category=FutureWarning) 

# --- 1. Constants and Shared Utilities ---
FS = 16000
D = 0.04
C = 343.0
ANGLE_TARGET = 90.0

# Training Settings
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 
N_FFT = 512
N_HOP = 256

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Core Physics Functions (from world.py)
def calculate_far_field_delays(azimuth_deg, d, c):
    theta_rad = np.deg2rad(azimuth_deg)
    tau_m1 = (d / 2) * np.cos(theta_rad - 0) / c
    tau_m2 = (d / 2) * np.cos(theta_rad - np.pi) / c
    return tau_m1, tau_m2

def apply_frac_delay(y, delay_sec, fs):
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    y_delayed = np.fft.irfft(y_fft * phase_shift, n=n)
    return y_delayed

def load_and_resample(file_path, target_fs):
    y, orig_fs = sf.read(file_path, dtype='float32')
    if len(y.shape) > 1: y = np.mean(y, axis=1)
    if orig_fs != target_fs: y = librosa.resample(y, orig_sr=orig_fs, target_sr=target_fs)
    return y

# --- 2. The Dataset (Generates Oracle Samples) ---
class SpatialAudioDataset(Dataset):
    def __init__(self, wav_files, segment_len_sec=2.0):
        self.wav_files = wav_files
        self.segment_len = int(segment_len_sec * FS)
        self.target_azimuth = ANGLE_TARGET
        self.interferer_azimuths = [40.0, 130.0]
        # Calculate expected STFT shape for model initialization
        self.n_freqs = len(scipy.signal.stft(np.zeros(self.segment_len), fs=FS, nperseg=N_FFT, noverlap=N_HOP)[0])
        self.n_time_frames = len(scipy.signal.stft(np.zeros(self.segment_len), fs=FS, nperseg=N_FFT, noverlap=N_HOP)[1])
        
    def __len__(self): return 5000 # Generate 5000 samples for training

    def __getitem__(self, idx):
        # On-the-fly mixing logic (simplified for speed)
        files = random.sample(self.wav_files, 3)
        audio_list = []
        for file in files:
            y = load_and_resample(file, FS)
            
            # --- FIX FOR VALUE ERROR ---
            if len(y) < self.segment_len:
                segment = np.pad(y, (0, self.segment_len - len(y)))
            else:
                start_idx = random.randint(0, len(y) - self.segment_len)
                segment = y[start_idx:start_idx + self.segment_len]
            
            audio_list.append(segment)
        # --- END FIX ---

        y_m1, y_m2 = np.zeros(self.segment_len), np.zeros(self.segment_len)
        s_target_clean, s_interferer_total = np.zeros(self.segment_len), np.zeros(self.segment_len)
        
        # Mix sources at defined angles
        for i, angle in enumerate([self.target_azimuth] + self.interferer_azimuths):
            s_clean = audio_list[i]
            tau_m1, tau_m2 = calculate_far_field_delays(angle, D, C)
            s_m1 = apply_frac_delay(s_clean, tau_m1, FS)
            
            y_m1 += s_m1
            y_m2 += apply_frac_delay(s_clean, tau_m2, FS)
            
            if i == 0: s_target_clean = s_m1 
            else: s_interferer_total += s_m1

        y_mix = np.stack([y_m1, y_m2], axis=0) # [2, Samples]
        
        # Compute STFTs for Inputs and Oracle Mask
        f, t, Y_mix = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
        _, _, S_tgt = scipy.signal.stft(s_target_clean, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
        _, _, S_int = scipy.signal.stft(s_interferer_total, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
        
        # --- INPUT FEATURES (Log-Mag and IPD) ---
        mag = np.abs(Y_mix)
        ipd = np.angle(Y_mix[0, :, :]) - np.angle(Y_mix[1, :, :])
        model_input = np.stack([np.log(mag[0, :, :] + 1e-7), ipd], axis=-1)
        
        # --- ORACLE MASK (The Answer Key) ---
        mask_target_oracle = np.where(np.abs(S_tgt) > np.abs(S_int), 1.0, 0.0)

        # Convert to PyTorch Tensors
        X = torch.from_numpy(model_input).float().permute(2, 0, 1) # [Features, Freq, Time]
        Y = torch.from_numpy(mask_target_oracle).float()

        return X, Y

# --- 3. The Neural Network (SHALLOW CNN: Memory Efficient) ---
class ShallowCNNMaskEstimator(nn.Module):
    def __init__(self, n_freqs, n_time_frames, n_features=2):
        super(ShallowCNNMaskEstimator, self).__init__()
        self.n_freqs = n_freqs
        self.n_time_frames = n_time_frames
        
        # Output size is the Freq x Time mask size
        out_f = n_freqs 
        out_t = n_time_frames

        # Sequential CNN structure (no memory-intensive skip connections)
        self.model = nn.Sequential(
            # Input: [B, 2, Freq, Time]
            nn.Conv2d(n_features, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Final 1x1 convolution maps 32 channels to 1 mask channel
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid() 
            # Output: [B, 1, Freq, Time]
        )

    def forward(self, x):
        # x shape: [Batch, Features, Freq, Time]
        x = self.model(x)
        # Squeeze channel dimension to match target shape [Batch, Freq, Time]
        return x.squeeze(1)

# --- 4. Training Loop ---
def train_model():
    print(f"--- 2. Neural Mask Estimator Training (SHALLOW CNN, Device: {device}) ---")
    
    # 1. Get Wav Files (Same reliable data loading)
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
    except:
        print("Fallback: Attempting to locate dataset in cache...")
        home_dir = os.path.expanduser("~")
        path = os.path.join(home_dir, ".cache", "kagglehub", "datasets", 
                            "mathurinache", "the-lj-speech-dataset", "versions", "1")
        if not os.path.exists(path):
             print("CRITICAL ERROR: Data download failed. Neither API nor cache found.")
             return
             
    wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
    all_wav_files = glob.glob(os.path.join(wav_path, "*.wav"))

    if not all_wav_files:
        print(f"CRITICAL ERROR: Found cache path {path} but no .wav files in {wav_path}")
        return

    # 2. Initialize Model Parameters
    temp_dataset = SpatialAudioDataset(all_wav_files[:5])
    n_freqs = temp_dataset.n_freqs
    n_time_frames = temp_dataset.n_time_frames

    # 3. Setup Model, Loss, and Optimizer
    # --- GPU INTEGRATION ---
    # USE THE SHALLOW CNN
    model = ShallowCNNMaskEstimator(n_freqs, n_time_frames).to(device)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    dataset = SpatialAudioDataset(all_wav_files)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 

    print(f"Starting training for {NUM_EPOCHS} epochs on SHALLOW CNN architecture.")
    
    # 4. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        # --- TQDM INTEGRATION ---
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for i, (inputs, targets) in enumerate(loop):
            # --- GPU INTEGRATION: Move data to device ---
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update TQDM Progress Bar
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
        
    # 5. Save Model
    torch.save(model.state_dict(), 'mask_estimator.pth')
    print("Training complete. Model saved to 'mask_estimator.pth'.")

if __name__ == "__main__":
    train_model()