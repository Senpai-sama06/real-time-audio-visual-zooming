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
import soundfile as sf # Needed for saving wavs
import pyroomacoustics as pra
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
from IPython.display import clear_output

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

# Load config or defaults
if os.path.exists("config.json"):
    import json
    with open("config.json", "r") as f: CONF = json.load(f)
else:
    # Fallback defaults
    print("[Warning] config.json not found, using defaults.")
    CONF = {
        "fs": 16000, "n_fft": 1024, "hop_len": 512, "train_seg_samples": 32000,
        "freq_cutoff_hz": 500, "low_freq_weight": 0.1, "high_freq_weight": 1.0,
        "room_dim": [4.9, 4.9, 4.9], "rt60_target": 0.5, "sir_target_db": 0,
        "mic_locs": [[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]], "mic_dist": 0.08, "n_interferers": 10
    }

FREQ_BINS = (CONF["n_fft"] // 2) + 1
INPUT_CHANNELS = 5
FIXED_TIME_STEPS = 64
BATCH_SIZE = 45
EPOCHS = 150
PATIENCE = 5
NUM_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.75))

# --- 1. Audio Saving Helper (NEW) ---
def save_debug_audio(raw_mix_stft, raw_target_stft, true_mask, pred_mask, epoch, save_dir="debug_audio"):
    """
    Reconstructs audio from STFT and saves debug files for listening.
    """
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
   
    # Unpack first sample from batch & move to CPU numpy
    # Shapes: Mix=(2, F, T), Tgt=(1, F, T), Masks=(1, F, T)
    Z_mix = raw_mix_stft[0].detach().cpu().numpy()
    Z_tgt = raw_target_stft[0].detach().cpu().numpy()
    M_true = true_mask[0].detach().cpu().numpy()
    M_pred = pred_mask[0].detach().cpu().numpy()

    # ISTFT Parameters (Must match training config)
    fs = CONF["fs"]
    n_fft = CONF["n_fft"]
    hop = CONF["hop_len"]
   
    def istft_stereo(stft_block):
        # stft_block shape: (2, F, T) -> returns (Samples, 2)
        _, t1 = scipy.signal.istft(stft_block[0], fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
        _, t2 = scipy.signal.istft(stft_block[1], fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
        # Normalize
        audio = np.stack([t1, t2], axis=1)
        return audio / (np.max(np.abs(audio)) + 1e-9)

    def istft_mono(stft_block):
        # stft_block shape: (1, F, T) or (F, T)
        if stft_block.ndim == 3: stft_block = stft_block[0]
        _, t = scipy.signal.istft(stft_block, fs=fs, nperseg=n_fft, noverlap=n_fft-hop)
        return t / (np.max(np.abs(t)) + 1e-9)

    # 1. Raw Input (Stereo)
    sf.write(f"{epoch_dir}/raw_audio.wav", istft_stereo(Z_mix), fs)
   
    # 2. Target Clean (Mono)
    sf.write(f"{epoch_dir}/target_audio.wav", istft_mono(Z_tgt), fs)
   
    # 3. Mask * Raw (Truth) -> Stereo
    # Apply mask to both channels
    Z_masked_true = Z_mix * M_true
    sf.write(f"{epoch_dir}/mask_truth_audio.wav", istft_stereo(Z_masked_true), fs)
   
    # 4. Mask * Raw (Predicted) -> Stereo
    Z_masked_pred = Z_mix * M_pred
    sf.write(f"{epoch_dir}/mask_pred_audio.wav", istft_stereo(Z_masked_pred), fs)
   
    print(f"[Debug] Saved audio samples to {epoch_dir}/")

# --- 2. Interactive Visualization Function ---
def visualize_masks(input_feat, true_mask, pred_mask, epoch):
    spec = input_feat[0, 0].detach().cpu().numpy() # Log Mag
    truth = true_mask[0, 0].detach().cpu().numpy()
    pred = pred_mask[0, 0].detach().cpu().numpy()
   
    clear_output(wait=True)
    plt.figure(figsize=(12, 4))
   
    plt.subplot(1, 3, 1)
    plt.title(f"Noisy Input (Ep {epoch})")
    plt.imshow(spec, aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
   
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(truth, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
   
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
   
    plt.tight_layout()
    # plt.show()
    plt.pause(0.1)

# --- 3. Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, path='best_model_phy.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'\n[EarlyStopping] Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        print(f'\n[EarlyStopping] Loss decreased. Model saved to {self.path}')

# --- 4. Dataset Generator (Updated Return) ---
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
        max_interf = CONF["n_interferers"]
        n_interf = random.randint(1, max_interf)
        total_sources = 1 + n_interf
       
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

        snr1 = random.choice(self.snr_options)
        snr2 = random.choice(self.snr_options)
       
        final_ch1 = self.add_awgn(clean_mix_ch1, snr1)
        final_ch2 = self.add_awgn(clean_mix_ch2, snr2)

        stereo_mix = np.stack([final_ch1, final_ch2])
        peak = np.max(np.abs(stereo_mix)) + 1e-9
       
        final_ch1 /= peak
        final_ch2 /= peak
        target_ch1 /= peak

        return final_ch1, final_ch2, target_ch1, interf_ch1_total

    def __getitem__(self, index):
        m1, m2, tgt_clean, int_sig = self.generate_simulation()

        _, _, Z = scipy.signal.stft(np.stack([m1, m2]), fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
        _, _, S_t = scipy.signal.stft(tgt_clean, fs=self.fs, nperseg=self.n_fft, noverlap=self.n_fft-self.hop)
       
        S_i = Z[0] - S_t

        mag = np.abs(Z)
        log_mag = np.log(mag[0] + 1e-7)
        raw_ipd = np.angle(Z[0]) - np.angle(Z[1])
        sin_ipd = np.sin(raw_ipd); cos_ipd = np.cos(raw_ipd)

        cross_spec = Z[0] * np.conj(Z[1])
        auto_spec0 = Z[0] * np.conj(Z[0])
        auto_spec1 = Z[1] * np.conj(Z[1])
        msc = np.abs(cross_spec) / (np.sqrt(np.abs(auto_spec0) * np.abs(auto_spec1)) + 1e-9)

        curr_t = log_mag.shape[1]
       
        # Pad Everything (Masks, Features, AND Raw Complex STFTs)
        pad = 0
        if curr_t < FIXED_TIME_STEPS:
            pad = FIXED_TIME_STEPS - curr_t
           
        def pad_tensor(t):
            # t shape: (..., time)
            if pad > 0:
                p = [(0, 0)] * (t.ndim - 1) + [(0, pad)]
                return np.pad(t, p)
            return t[:, :FIXED_TIME_STEPS] if t.shape[-1] > FIXED_TIME_STEPS else t

        log_mag = pad_tensor(log_mag)
        sin_ipd = pad_tensor(sin_ipd)
        cos_ipd = pad_tensor(cos_ipd)
        msc = pad_tensor(msc)
       
        # Pad Raw Complex STFTs (for reconstruction)
        Z_padded = pad_tensor(Z)
        St_padded = pad_tensor(S_t[np.newaxis, ...]) # make it (1, F, T) for uniform padding logic

        t_steps = FIXED_TIME_STEPS
        f_map_tiled = np.tile(self.freq_map, (1, t_steps))
       
        feat = np.stack([log_mag, sin_ipd, cos_ipd, f_map_tiled, msc], axis=-1)
        feat = np.transpose(feat, (2, 0, 1))

        mag_t = np.abs(S_t); mag_i = np.abs(S_i)
        mag_t = pad_tensor(mag_t)
        mag_i = pad_tensor(mag_i)

        irm = mag_t / (mag_t + mag_i + 1e-7)
        irm = irm[np.newaxis, ...]
       
        # Return extra raw complex data
        return (torch.from_numpy(feat).float(),
                torch.from_numpy(irm).float(),
                torch.from_numpy(mag_t).float(),
                torch.from_numpy(Z_padded).cfloat(),
                torch.from_numpy(St_padded).cfloat())

# --- 5. Main ---
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

    if len(files) < 10: return

    random.shuffle(files)
    half_point = int(0.9*len(files))
    training_pool = files[:half_point]
    unused_pool = files[half_point:]
    val_files = random.sample(unused_pool, min(len(unused_pool), 200))

    train_ds = PhysicsAudioDataset(training_pool, is_validation=False)
    val_ds = PhysicsAudioDataset(val_files, is_validation=True)
   
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # Need to redefine model class since not imported (assumed same as previous)
    # Just reusing the DeepFPU from previous context for completeness of this file
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)); self.relu = nn.ReLU()
        def forward(self, x): return self.relu(x + self.conv(x))

    class DeepFPU(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=(1, 2))
            self.enc1_conv = nn.Sequential(nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            self.enc2_conv = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64))
            self.enc3_conv = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128))
            self.enc4_conv = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256))
            self.bottleneck = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), ResBlock(512), ResBlock(512))
            self.up4 = nn.ConvTranspose2d(512, 256, (1, 2), stride=(1, 2)); self.dec4_conv = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), ResBlock(256))
            self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2)); self.dec3_conv = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResBlock(128))
            self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2)); self.dec2_conv = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64))
            self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2)); self.dec1_conv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            self.out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        def _match(self, x, target):
            if x.shape[3] != target.shape[3]: x = F.interpolate(x, size=target.shape[2:], mode='nearest')
            return x
        def forward(self, x):
            e1 = self.enc1_conv(x); e2 = self.enc2_conv(self.pool(e1)); e3 = self.enc3_conv(self.pool(e2)); e4 = self.enc4_conv(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            u4 = self._match(self.up4(b), e4); d4 = self.dec4_conv(torch.cat([u4, e4], dim=1))
            u3 = self._match(self.up3(d4), e3); d3 = self.dec3_conv(torch.cat([u3, e3], dim=1))
            u2 = self._match(self.up2(d3), e2); d2 = self.dec2_conv(torch.cat([u2, e2], dim=1))
            u1 = self._match(self.up1(d2), e1); d1 = self.dec1_conv(torch.cat([u1, e1], dim=1))
            return self.out(d1)

    model = DeepFPU().to(DEVICE)
    loss_fn = nn.BCELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopper = EarlyStopping(patience=PATIENCE)    

    print(f"[PyTorch] Starting Training on {DEVICE}...")
   
    debug_batch = None
   
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
       
        # Unpack 5 items now: x, y, mag_t, Z_mix, Z_target
        for batch_idx, (x, y, mag_t, _, _) in enumerate(loop):
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
            for batch_idx, (x, y, mag_t, Z_mix, Z_target) in enumerate(val_loader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
                if batch_idx == 0:
                    debug_batch = (x, y, pred, Z_mix, Z_target)
       
        # Save Debug Audio
        if debug_batch is not None:
             # debug_batch index: 0=x, 1=y(truth), 2=pred, 3=Z_mix, 4=Z_target
            #  visualize_masks(debug_batch[0], debug_batch[1], debug_batch[2], epoch + 1)
             save_debug_audio(debug_batch[3], debug_batch[4], debug_batch[1], debug_batch[2], epoch+1)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
       
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Results | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {current_lr:.6f}")
       
        scheduler.step(avg_train)
        early_stopper(avg_train, model)
        if early_stopper.early_stop:
            print("[EarlyStopping] Triggered. Stopping.")
            break

    torch.save(model.state_dict(), "mask_estimator_phy_final.pth")
    print("Done.")

if __name__ == "__main__":
    main()
