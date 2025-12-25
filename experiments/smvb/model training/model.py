# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import os
# import glob
# import scipy.signal
# import librosa
# import random
# import sys
# import pyroomacoustics as pra
# from tqdm import tqdm
# import multiprocessing

# # --- 0. Device & Configuration ---
# def get_device():
#     if torch.cuda.is_available():
#         d = torch.device("cuda")
#         print(f"[System] Using GPU: {torch.cuda.get_device_name(0)}")
#         return d
#     elif torch.backends.mps.is_available():
#         print("[System] Using Apple MPS (Metal)")
#         return torch.device("mps")
#     else:
#         print("[System] WARNING: Using CPU (Training will be extremely slow)")
#         return torch.device("cpu")

# DEVICE = get_device()

# CONF = {
#     "fs": 16000,
#     "n_fft": 1024,
#     "hop_len": 512,
#     "train_seg_samples": 32000,
#     "freq_cutoff_hz": 500,
#     "low_freq_weight": 0.1,
#     "high_freq_weight": 1.0,
   
#     "room_dim": [4.9, 4.9, 4.9],
#     "rt60_target": 0.5,
#     "snr_target": 5,            
#     "sir_target_db": 0,          
   
#     "mic_locs": [[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]],
#     "mic_dist": 0.08,
#     "n_interferers": 3
# }

# FREQ_BINS = (CONF["n_fft"] // 2) + 1
# INPUT_CHANNELS = 4
# FIXED_TIME_STEPS = 64
# BATCH_SIZE = 16
# EPOCHS = 100
# PATIENCE = 5
# NUM_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.75))

# # --- 1. Early Stopping Mechanism ---
# class EarlyStopping:
#     def __init__(self, patience=5, min_delta=0.001, path='best_model_phy.pth'):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.path = path
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False

#     def __call__(self, val_loss, model):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             self.save_checkpoint(model)
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             print(f'\n[EarlyStopping] Counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.save_checkpoint(model)
#             self.counter = 0

#     def save_checkpoint(self, model):
#         torch.save(model.state_dict(), self.path)
#         print(f'\n[EarlyStopping] Validation loss decreased. Model saved to {self.path}')

# # --- 2. Advanced Physics Dataset ---
# class PhysicsAudioDataset(Dataset):
#     def __init__(self, wav_files, is_validation=False):
#         self.wav_files = wav_files
#         self.is_validation = is_validation
#         self.fs = CONF["fs"]
#         self.seg_len = CONF["train_seg_samples"]
#         self.n_fft = CONF["n_fft"]
#         self.hop = CONF["hop_len"]
#         self.freq_map = np.linspace(0, 1, FREQ_BINS, dtype=np.float32)[:, np.newaxis]
       
#         self.room_dim = CONF["room_dim"]
#         self.mic_locs = np.array(CONF["mic_locs"]).T
#         self.rt60 = CONF["rt60_target"]
#         self.snr_target = CONF["snr_target"]
#         self.sir_target_db = CONF["sir_target_db"]

#     def __len__(self):
#         return len(self.wav_files)

#     def add_awgn(self, signal, snr_db):
#         sig_power = np.mean(signal ** 2)
#         if sig_power == 0: return signal
#         noise_power = sig_power / (10 ** (snr_db / 10))
#         noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
#         return signal + noise

#     def generate_simulation(self):
#         # Sample 1 Target + N Interferers
#         n_interf = CONF["n_interferers"]
#         total_sources = 1 + n_interf
       
#         if len(self.wav_files) < total_sources:
#              samples = random.choices(self.wav_files, k=total_sources)
#         else:
#              samples = random.sample(self.wav_files, total_sources)
           
#         sigs = []
#         for s in samples:
#             y, _ = librosa.load(s, sr=self.fs, mono=True)
#             while len(y) < self.seg_len: y = np.concatenate([y, y])
#             start = random.randint(0, len(y) - self.seg_len)
#             sigs.append(y[start:start+self.seg_len])
           
#         target_sig = sigs[0]
#         interf_sigs = sigs[1:]

#         # Setup Room
#         e_absorption, _ = pra.inverse_sabine(self.rt60, self.room_dim)
#         materials = pra.Material(e_absorption)
#         room = pra.ShoeBox(self.room_dim, fs=self.fs, materials=materials, max_order=3)
#         room.add_microphone_array(self.mic_locs)

#         # Sources
#         room.add_source([2.45, 3.45, 1.5], signal=target_sig)
#         for i_sig in interf_sigs:
#             rx = random.uniform(1.0, self.room_dim[0]-1.0)
#             ry = random.uniform(1.0, self.room_dim[1]-1.0)
#             room.add_source([rx, ry, 1.5], signal=i_sig)

#         room.compute_rir()
       
#         def get_conv(s, h):
#             return scipy.signal.fftconvolve(s, h, mode='full')[:self.seg_len]

#         target_ch1 = get_conv(target_sig, room.rir[0][0])
#         target_ch2 = get_conv(target_sig, room.rir[1][0])
       
#         interf_ch1_total = np.zeros(self.seg_len)
#         interf_ch2_total = np.zeros(self.seg_len)

#         for i, i_sig in enumerate(interf_sigs):
#             src_idx = i + 1
#             i_ch1 = get_conv(i_sig, room.rir[0][src_idx])
#             i_ch2 = get_conv(i_sig, room.rir[1][src_idx])
#             interf_ch1_total += i_ch1
#             interf_ch2_total += i_ch2

#         p_target = np.mean(target_ch1 ** 2)
#         p_interf = np.mean(interf_ch1_total ** 2)
       
#         if p_interf > 0:
#             desired_ratio = 10**(self.sir_target_db/10)
#             gain = np.sqrt(p_target / (p_interf * desired_ratio))
#             interf_ch1_total *= gain
#             interf_ch2_total *= gain

#         clean_mix_ch1 = target_ch1 + interf_ch1_total
#         clean_mix_ch2 = target_ch2 + interf_ch2_total

#         final_ch1 = self.add_awgn(clean_mix_ch1, self.snr_target)
#         final_ch2 = self.add_awgn(clean_mix_ch2, self.snr_target)

#         stereo_mix = np.stack([final_ch1, final_ch2])
#         peak = np.max(np.abs(stereo_mix)) + 1e-9
       
#         final_ch1 /= peak
#         final_ch2 /= peak
#         target_ch1 /= peak

#         return final_ch1, final_ch2, target_ch1, interf_ch1_total

#     def __getitem__(self, index):
#         m1, m2, tgt, int_sig = self.generate_simulation()

#         _, _, Z = scipy.signal.stft(np.stack([m1, m2]), fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
#         _, _, S_t = scipy.signal.stft(tgt, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
#         _, _, S_i = scipy.signal.stft(int_sig, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)

#         mag = np.abs(Z)
#         log_mag = np.log(mag[0] + 1e-7)
#         raw_ipd = np.angle(Z[0]) - np.angle(Z[1])
#         sin_ipd = np.sin(raw_ipd); cos_ipd = np.cos(raw_ipd)

#         curr_t = log_mag.shape[1]
#         if curr_t < FIXED_TIME_STEPS:
#             pad = FIXED_TIME_STEPS - curr_t
#             log_mag = np.pad(log_mag, ((0,0),(0,pad)))
#             sin_ipd = np.pad(sin_ipd, ((0,0),(0,pad)))
#             cos_ipd = np.pad(cos_ipd, ((0,0),(0,pad)))
#             t_steps = FIXED_TIME_STEPS
#         else:
#             log_mag = log_mag[:, :FIXED_TIME_STEPS]
#             sin_ipd = sin_ipd[:, :FIXED_TIME_STEPS]
#             cos_ipd = cos_ipd[:, :FIXED_TIME_STEPS]
#             t_steps = FIXED_TIME_STEPS

#         f_map_tiled = np.tile(self.freq_map, (1, t_steps))
       
#         feat = np.stack([log_mag, sin_ipd, cos_ipd, f_map_tiled], axis=-1)
#         feat = np.transpose(feat, (2, 0, 1))

#         mag_t = np.abs(S_t); mag_i = np.abs(S_i)
#         if mag_t.shape[1] < FIXED_TIME_STEPS:
#             pad = FIXED_TIME_STEPS - mag_t.shape[1]
#             mag_t = np.pad(mag_t, ((0,0),(0,pad)))
#             mag_i = np.pad(mag_i, ((0,0),(0,pad)))
#         else:
#             mag_t = mag_t[:, :FIXED_TIME_STEPS]
#             mag_i = mag_i[:, :FIXED_TIME_STEPS]

#         irm = mag_t / (mag_t + mag_i + 1e-7)
#         irm = irm[np.newaxis, ...]

#         return torch.from_numpy(feat).float(), torch.from_numpy(irm).float()

# # --- 3. Model & Loss ---
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, filters):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, filters, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(filters)
#         self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(filters)
#         self.relu = nn.ReLU()
#         self.shortcut = nn.Conv2d(in_channels, filters, 1) if in_channels != filters else nn.Identity()
#     def forward(self, x):
#         return self.relu(self.conv2(self.relu(self.bn1(self.conv1(x)))) + self.shortcut(x))

# class ComplexUNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.e1 = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
#                                 nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
#         self.e2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64, 64))
#         self.e3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128, 128))
#         self.e4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256, 256))
#         self.pool = nn.MaxPool2d((1, 2))

#         self.b = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
#                                ResBlock(512, 512), ResBlock(512, 512))
       
#         self.up4 = nn.ConvTranspose2d(512, 256, (1, 2), stride=(1, 2))
#         self.d4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256, 256))
#         self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
#         self.d3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128, 128))
#         self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
#         self.d2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64, 64))
#         self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2))
#         self.d1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
#                                 nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
#         self.out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

#     def forward(self, x):
#         e1 = self.e1(x); p1 = self.pool(e1)
#         e2 = self.e2(p1); p2 = self.pool(e2)
#         e3 = self.e3(p2); p3 = self.pool(e3)
#         e4 = self.e4(p3); p4 = self.pool(e4)
#         b = self.b(p4)
#         u4 = F.interpolate(self.up4(b), size=e4.shape[2:])
#         d4 = self.d4(torch.cat([u4, e4], 1))
#         u3 = F.interpolate(self.up3(d4), size=e3.shape[2:])
#         d3 = self.d3(torch.cat([u3, e3], 1))
#         u2 = F.interpolate(self.up2(d3), size=e2.shape[2:])
#         d2 = self.d2(torch.cat([u2, e2], 1))
#         u1 = F.interpolate(self.up1(d2), size=e1.shape[2:])
#         d1 = self.d1(torch.cat([u1, e1], 1))
#         return self.out(d1)

# class PhysicsWeightedLoss(nn.Module):
#     def __init__(self, fs, n_fft, cutoff_hz=500, w_low=0.1, w_high=1.0):
#         super().__init__()
#         freqs = np.fft.rfftfreq(n_fft, 1/fs)
#         weights = np.where(freqs < cutoff_hz, w_low, w_high)
#         self.W = torch.tensor(weights.reshape(1, 1, -1, 1), dtype=torch.float32)

#     def forward(self, y_pred, y_true):
#         if self.W.device != y_pred.device: self.W = self.W.to(y_pred.device)
#         return torch.mean(torch.square(y_true - y_pred) * self.W)

# # --- 4. Main ---
# def main():
#     try:
#         import kagglehub
#         path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
#         wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
#         files = glob.glob(os.path.join(wav_path, "*.wav"))
#     except:
#         home = os.path.expanduser("~")
#         wav_path = os.path.join(home, ".cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset/versions/1/LJSpeech-1.1/wavs")
#         files = glob.glob(os.path.join(wav_path, "*.wav"))

#     if len(files) < 10:
#         print("Error: Files not found.")
#         return

#     random.shuffle(files)
   
#     # 30% of data for training (faster experimentation)
#     half_point = int(0.3 * len(files))
#     training_pool = files[:half_point]
#     unused_pool = files[half_point:]
   
#     val_files = random.sample(unused_pool, min(len(unused_pool), 200))

#     print(f"[PyTorch] Total Files: {len(files)}")
#     print(f"[PyTorch] Training Pool (30%): {len(training_pool)}")
#     print(f"[PyTorch] Workers: {NUM_WORKERS}")

#     train_ds = PhysicsAudioDataset(training_pool, is_validation=False)
#     val_ds = PhysicsAudioDataset(val_files, is_validation=True)
   
#     # OPTIMIZED DATALOADERS
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,    # Parallel loading
#         pin_memory=True,            # Faster CPU->GPU transfer
#         persistent_workers=True,    # Keep workers alive
#         prefetch_factor=2           # Buffer batches
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True,
#         persistent_workers=True,
#         prefetch_factor=2
#     )

#     model = ComplexUNet().to(DEVICE)
#     loss_fn = PhysicsWeightedLoss(CONF["fs"], CONF["n_fft"]).to(DEVICE)
   
#     # --- UPDATED: Optimizer with Regularization (Weight Decay) ---
#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
   
#     # --- UPDATED: Learning Rate Scheduler ---
#     # Reduce LR by factor of 0.5 if validation loss doesn't improve for 2 epochs
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=2, verbose=True
#     )
   
#     early_stopper = EarlyStopping(patience=PATIENCE)

#     print(f"[PyTorch] Starting Training on {DEVICE}...")
   
   

#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0.0
       
#         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
       
#         for batch_idx, (x, y) in enumerate(loop):
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#             pred = model(x)
#             loss = loss_fn(pred, y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
       
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(DEVICE), y.to(DEVICE)
#                 pred = model(x)
#                 val_loss += loss_fn(pred, y).item()
       
#         avg_train = train_loss / len(train_loader)
#         avg_val = val_loss / len(val_loader)
       
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"Epoch {epoch+1} Results | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {current_lr:.6f}")
       
#         # --- UPDATED: Step the Scheduler ---
#         scheduler.step(avg_val)
       
#         early_stopper(avg_val, model)
#         if early_stopper.early_stop:
#             print("[EarlyStopping] Triggered. Stopping.")
#             break

#     torch.save(model.state_dict(), "mask_estimator_phy_final.pth")
#     print("Done.")

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import scipy.signal
import librosa
import random
import sys
import pyroomacoustics as pra
from tqdm import tqdm
import multiprocessing

# --- 0. Device & Configuration ---
def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"[System] Using GPU: {torch.cuda.get_device_name(0)}")
        return d
    elif torch.backends.mps.is_available():
        print("[System] Using Apple MPS (Metal)")
        return torch.device("mps")
    else:
        print("[System] WARNING: Using CPU (Training will be extremely slow)")
        return torch.device("cpu")

DEVICE = get_device()

CONF = {
    "fs": 16000,
    "n_fft": 1024,
    "hop_len": 512,
    "train_seg_samples": 32000,
    "freq_cutoff_hz": 500,
    "low_freq_weight": 0.1,
    "high_freq_weight": 1.0,
   
    "room_dim": [4.9, 4.9, 4.9],
    "rt60_target": 0.5,
    # "snr_target" REMOVED: Now handled dynamically
    "sir_target_db": 0,          
   
    "mic_locs": [[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]],
    "mic_dist": 0.08,
    "n_interferers": 10
}

FREQ_BINS = (CONF["n_fft"] // 2) + 1
INPUT_CHANNELS = 4
FIXED_TIME_STEPS = 64
BATCH_SIZE = 50
EPOCHS = 100
PATIENCE = 5
NUM_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.75))

# --- 1. Early Stopping Mechanism ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, path='best_model_phy.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'\n[EarlyStopping] Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f'\n[EarlyStopping] Validation loss decreased. Model saved to {self.path}')

# --- 2. Advanced Physics Dataset ---
class PhysicsAudioDataset(Dataset):
    def __init__(self, wav_files, is_validation=False):
        self.wav_files = wav_files
        self.is_validation = is_validation
        self.fs = CONF["fs"]
        self.seg_len = CONF["train_seg_samples"]
        self.n_fft = CONF["n_fft"]
        self.hop = CONF["hop_len"]
        self.freq_map = np.linspace(0, 1, FREQ_BINS, dtype=np.float32)[:, np.newaxis]
       
        self.room_dim = CONF["room_dim"]
        self.mic_locs = np.array(CONF["mic_locs"]).T
        self.rt60 = CONF["rt60_target"]
        self.sir_target_db = CONF["sir_target_db"]
       
        # Define the SNR choices
        self.snr_options = [5, 10, 15, 25]

    def __len__(self):
        return len(self.wav_files)

    def add_awgn(self, signal, snr_db):
        sig_power = np.mean(signal ** 2)
        if sig_power == 0: return signal
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise

    def generate_simulation(self):
        # 1. Randomize Number of Interferers
        max_interf = CONF["n_interferers"]
        n_interf = random.randint(1, max_interf)
        total_sources = 1 + n_interf
       
        # 2. Pick Samples
        if len(self.wav_files) < total_sources:
             samples = random.choices(self.wav_files, k=total_sources)
        else:
             samples = random.sample(self.wav_files, total_sources)
           
        sigs = []
        for s in samples:
            y, _ = librosa.load(s, sr=self.fs, mono=True)
            while len(y) < self.seg_len: y = np.concatenate([y, y])
            start = random.randint(0, len(y) - self.seg_len)
            sigs.append(y[start:start+self.seg_len])
           
        target_sig = sigs[0]
        interf_sigs = sigs[1:]

        # 3. Setup Room
        e_absorption, _ = pra.inverse_sabine(self.rt60, self.room_dim)
        materials = pra.Material(e_absorption)
        room = pra.ShoeBox(self.room_dim, fs=self.fs, materials=materials, max_order=3)
        room.add_microphone_array(self.mic_locs)

        room.add_source([2.45, 3.45, 1.5], signal=target_sig)
        for i_sig in interf_sigs:
            rx = random.uniform(1.0, self.room_dim[0]-1.0)
            ry = random.uniform(1.0, self.room_dim[1]-1.0)
            room.add_source([rx, ry, 1.5], signal=i_sig)

        room.compute_rir()
       
        def get_conv(s, h):
            return scipy.signal.fftconvolve(s, h, mode='full')[:self.seg_len]

        target_ch1 = get_conv(target_sig, room.rir[0][0])
        target_ch2 = get_conv(target_sig, room.rir[1][0])
       
        interf_ch1_total = np.zeros(self.seg_len)
        interf_ch2_total = np.zeros(self.seg_len)

        for i, i_sig in enumerate(interf_sigs):
            src_idx = i + 1
            i_ch1 = get_conv(i_sig, room.rir[0][src_idx])
            i_ch2 = get_conv(i_sig, room.rir[1][src_idx])
            interf_ch1_total += i_ch1
            interf_ch2_total += i_ch2

        p_target = np.mean(target_ch1 ** 2)
        p_interf = np.mean(interf_ch1_total ** 2)
       
        if p_interf > 0:
            desired_ratio = 10**(self.sir_target_db/10)
            gain = np.sqrt(p_target / (p_interf * desired_ratio))
            interf_ch1_total *= gain
            interf_ch2_total *= gain

        clean_mix_ch1 = target_ch1 + interf_ch1_total
        clean_mix_ch2 = target_ch2 + interf_ch2_total

        # 4. RANDOM SNR SELECTION
        # Pick a random SNR from the list for this specific sample
        current_snr = random.choice(self.snr_options)
       
        final_ch1 = self.add_awgn(clean_mix_ch1, current_snr)
        final_ch2 = self.add_awgn(clean_mix_ch2, current_snr)

        # STFT and Feature Extraction
        stereo_mix = np.stack([final_ch1, final_ch2])
        peak = np.max(np.abs(stereo_mix)) + 1e-9
       
        final_ch1 /= peak
        final_ch2 /= peak
        target_ch1 /= peak

        return final_ch1, final_ch2, target_ch1, interf_ch1_total

    def __getitem__(self, index):
        m1, m2, tgt, int_sig = self.generate_simulation()

        _, _, Z = scipy.signal.stft(np.stack([m1, m2]), fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
        _, _, S_t = scipy.signal.stft(tgt, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
        _, _, S_i = scipy.signal.stft(int_sig, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)

        mag = np.abs(Z)
        log_mag = np.log(mag[0] + 1e-7)
        raw_ipd = np.angle(Z[0]) - np.angle(Z[1])
        sin_ipd = np.sin(raw_ipd); cos_ipd = np.cos(raw_ipd)

        curr_t = log_mag.shape[1]
        if curr_t < FIXED_TIME_STEPS:
            pad = FIXED_TIME_STEPS - curr_t
            log_mag = np.pad(log_mag, ((0,0),(0,pad)))
            sin_ipd = np.pad(sin_ipd, ((0,0),(0,pad)))
            cos_ipd = np.pad(cos_ipd, ((0,0),(0,pad)))
            t_steps = FIXED_TIME_STEPS
        else:
            log_mag = log_mag[:, :FIXED_TIME_STEPS]
            sin_ipd = sin_ipd[:, :FIXED_TIME_STEPS]
            cos_ipd = cos_ipd[:, :FIXED_TIME_STEPS]
            t_steps = FIXED_TIME_STEPS

        f_map_tiled = np.tile(self.freq_map, (1, t_steps))
       
        feat = np.stack([log_mag, sin_ipd, cos_ipd, f_map_tiled], axis=-1)
        feat = np.transpose(feat, (2, 0, 1))

        mag_t = np.abs(S_t); mag_i = np.abs(S_i)
        if mag_t.shape[1] < FIXED_TIME_STEPS:
            pad = FIXED_TIME_STEPS - mag_t.shape[1]
            mag_t = np.pad(mag_t, ((0,0),(0,pad)))
            mag_i = np.pad(mag_i, ((0,0),(0,pad)))
        else:
            mag_t = mag_t[:, :FIXED_TIME_STEPS]
            mag_i = mag_i[:, :FIXED_TIME_STEPS]

        irm = mag_t / (mag_t + mag_i + 1e-7)
        irm = irm[np.newaxis, ...]

        return torch.from_numpy(feat).float(), torch.from_numpy(irm).float()

# --- 3. UPDATED Model Architecture (DeepFPU - Frequency Preserving) ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()
   
    def forward(self, x):
        return self.relu(x + self.conv(x))

class DeepFPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
       
        # -- Encoder --
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.enc2_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64)
        )
        self.enc3_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128)
        )
        self.enc4_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256)
        )
       
        # -- Bottleneck --
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            ResBlock(512),
            ResBlock(512)
        )
       
        # -- Decoder --
        self.up4 = nn.ConvTranspose2d(512, 256, (1, 2), stride=(1, 2))
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256)
        )
        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128)
        )
        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64)
        )
        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2))
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
       
        self.out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def _match(self, x, target):
        if x.shape[3] != target.shape[3]:
            x = F.interpolate(x, size=target.shape[2:], mode='nearest')
        return x

    def forward(self, x):
        e1 = self.enc1_conv(x)
        e2 = self.enc2_conv(self.pool(e1))
        e3 = self.enc3_conv(self.pool(e2))
        e4 = self.enc4_conv(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        u4 = self._match(self.up4(b), e4)
        d4 = self.dec4_conv(torch.cat([u4, e4], dim=1))
        u3 = self._match(self.up3(d4), e3)
        d3 = self.dec3_conv(torch.cat([u3, e3], dim=1))
        u2 = self._match(self.up2(d3), e2)
        d2 = self.dec2_conv(torch.cat([u2, e2], dim=1))
        u1 = self._match(self.up1(d2), e1)
        d1 = self.dec1_conv(torch.cat([u1, e1], dim=1))
        return self.out(d1)

class PhysicsWeightedLoss(nn.Module):
    def __init__(self, fs, n_fft, cutoff_hz=500, w_low=0.1, w_high=1.0):
        super().__init__()
        freqs = np.fft.rfftfreq(n_fft, 1/fs)
        weights = np.where(freqs < cutoff_hz, w_low, w_high)
        self.W = torch.tensor(weights.reshape(1, 1, -1, 1), dtype=torch.float32)

    def forward(self, y_pred, y_true):
        if self.W.device != y_pred.device: self.W = self.W.to(y_pred.device)
        return torch.mean(torch.square(y_true - y_pred) * self.W)

# --- 4. Main ---
def main():
    try:
        import kagglehub
        path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))
    except:
        home = os.path.expanduser("~")
        wav_path = os.path.join(home, ".cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset/versions/1/LJSpeech-1.1/wavs")
        files = glob.glob(os.path.join(wav_path, "*.wav"))

    if len(files) < 10:
        print("Error: Files not found.")
        return

    random.shuffle(files)
   
    #100% data
    half_point = int(0.9 * len(files))
    training_pool = files[:half_point]
    unused_pool = files[half_point:]
   
    val_files = random.sample(unused_pool, min(len(unused_pool), 200))

    print(f"[PyTorch] Total Files: {len(files)}")
    print(f"[PyTorch] Training Pool (50%): {len(training_pool)}")
    print(f"[PyTorch] Workers: {NUM_WORKERS}")

    train_ds = PhysicsAudioDataset(training_pool, is_validation=False)
    val_ds = PhysicsAudioDataset(val_files, is_validation=True)
   
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,            
        persistent_workers=True,    
        prefetch_factor=2          
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = DeepFPU().to(DEVICE)
    loss_fn = PhysicsWeightedLoss(CONF["fs"], CONF["n_fft"]).to(DEVICE)
   
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
   
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
   
    early_stopper = EarlyStopping(patience=PATIENCE)

    print(f"[PyTorch] Starting Training on {DEVICE}...")
   
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
       
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
       
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
       
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
       
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
       
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Results | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {current_lr:.6f}")
       
        scheduler.step(avg_val)
       
        early_stopper(avg_val, model)
        if early_stopper.early_stop:
            print("[EarlyStopping] Triggered. Stopping.")
            break

    torch.save(model.state_dict(), "mask_estimator_phy_final.pth")
    print("Done.")

if __name__ == "__main__":
    main()