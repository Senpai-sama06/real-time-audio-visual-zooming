import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.signal
import os
import random
import glob
import librosa
import json
import torch.nn.functional as F
from tqdm import tqdm

# --- 1. Load Config ---
if not os.path.exists("config.json"):
    raise FileNotFoundError("Run world_building.py first.")

with open("config.json", "r") as f:
    CONF = json.load(f)

FS = CONF["fs"]
N_FFT = CONF["n_fft"]
HOP = CONF["hop_len"]
SEGMENT_LEN = CONF["train_seg_samples"]
D = CONF["d"]
C = CONF["c"]

# Training Params
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
ANGLE_TARGET = 90.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Helpers ---
def calculate_far_field_delays(azimuth_deg, d, c):
    theta = np.deg2rad(azimuth_deg)
    return (d/2)*np.cos(theta)/c, (d/2)*np.cos(theta-np.pi)/c

def apply_frac_delay(y, delay_sec, fs):
    n = len(y)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    phase = np.exp(-1j * 2 * np.pi * freqs * delay_sec)
    return np.fft.irfft(y_fft * phase, n=n)

# --- 3. Dataset (On-the-Fly Mixing) ---
class SpatialDataset(Dataset):
    def __init__(self, files):
        self.files = files
        
    def __len__(self): return 3000

    def __getitem__(self, idx):
        selected = random.sample(self.files, 3)
        raw = []
        for f in selected:
            y, _ = librosa.load(f, sr=FS)
            if len(y) < SEGMENT_LEN: 
                y = np.pad(y, (0, SEGMENT_LEN - len(y)))
            start = random.randint(0, len(y) - SEGMENT_LEN)
            raw.append(y[start:start+SEGMENT_LEN])

        # Mix
        angles = [ANGLE_TARGET, 40.0, 130.0] # Target, Int A, Int B
        m1, m2, tgt, int_sig = np.zeros(SEGMENT_LEN), np.zeros(SEGMENT_LEN), np.zeros(SEGMENT_LEN), np.zeros(SEGMENT_LEN)
        
        for i, angle in enumerate(angles):
            t1, t2 = calculate_far_field_delays(angle, D, C)
            s1 = apply_frac_delay(raw[i], t1, FS)
            m1 += s1
            m2 += apply_frac_delay(raw[i], t2, FS)
            if i == 0: tgt = s1
            else: int_sig += s1

        y_mix = np.stack([m1, m2], axis=0)

        # STFT (Standard Hann Window)
        f, t, Y = scipy.signal.stft(y_mix, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)
        _, _, S_t = scipy.signal.stft(tgt, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)
        _, _, S_i = scipy.signal.stft(int_sig, fs=FS, nperseg=N_FFT, noverlap=N_FFT-HOP)

        # Features
        mag = np.abs(Y)
        ipd = np.angle(Y[0]) - np.angle(Y[1])
        feat = np.stack([np.log(mag[0]+1e-7), ipd], axis=0)
        
        # Label (Oracle Mask)
        mask = np.where(np.abs(S_t) > np.abs(S_i), 1.0, 0.0)

        return torch.from_numpy(feat).float(), torch.from_numpy(mask).float()

# --- 4. Freq-Preserving U-Net ---
class FreqPreservingUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2)) # Pool Time Only
        
        self.enc1 = self._conv(2, 32)
        self.enc2 = self._conv(32, 64)
        self.enc3 = self._conv(64, 128)
        self.bot = self._conv(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.dec3 = self._conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.dec2 = self._conv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2))
        self.dec1 = self._conv(64, 32)
        self.out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def _conv(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()
        )
    
    def _match(self, x, target):
        if x.shape[3] != target.shape[3]:
            x = F.interpolate(x, size=target.shape[2:], mode='nearest')
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        
        u3 = self._match(self.up3(b), e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1).squeeze(1)

# --- 5. Main ---
def main():
    # Load files logic
    try:
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
    except:
        home = os.path.expanduser("~")
        wav_path = os.path.join(home, ".cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset/versions/1/LJSpeech-1.1/wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))

    if not files: return

    model = FreqPreservingUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.BCELoss()
    loader = DataLoader(SpatialDataset(files), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    for ep in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Ep {ep+1}", leave=False)

        epoch_loss = 0
        total_samples = 0

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()

            # accumulate
            batch_size = x.size(0)
            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

            loop.set_postfix(loss=loss.item())

        mean_loss = epoch_loss / total_samples
        print(f"Epoch {ep+1}/{NUM_EPOCHS} — Mean Loss: {mean_loss:.6f}")    
    torch.save(model.state_dict(), "mask_estimator.pth")
    print("Saved mask_estimator.pth")

if __name__ == "__main__":
    main()