import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import h5py
from tqdm import tqdm
import scipy.signal as signal
from torch.optim.lr_scheduler import CosineAnnealingLR

# -----------------------------------------------------------------------------
# Configuration and Hyperparameters
# -----------------------------------------------------------------------------
INPUT_SAMPLING_RATE = 50000.0
WINDOW_DURATION = 0.020
STEP_DURATION = 0.01
WINDOW_SIZE = int(WINDOW_DURATION * INPUT_SAMPLING_RATE)
STEP_SIZE = int(STEP_DURATION * INPUT_SAMPLING_RATE)

MAX_TRAIN_SAMPLES = 40000
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ECE_CHANNELS_SELECTED = ['HRS03H', 'HRS34H', 'HRS33H', 'HRS01H', 'HRS04H', 'HRS37H',
                         'HRS40H', 'HRS05H', 'HRS06H', 'HRS07H', 'HRS12H', 'HRS14H']
MAG_CHANNELS = ['KHP7T', 'LHP7T']
ALL_CHANNELS = sorted(list(set(ECE_CHANNELS_SELECTED + MAG_CHANNELS)))

mag_indices = [ALL_CHANNELS.index(ch) for ch in MAG_CHANNELS]
spatial_ece_indices = [ALL_CHANNELS.index(ch) for ch in ECE_CHANNELS_SELECTED]
NUM_SPATIAL_ECE_CHANNELS = len(ECE_CHANNELS_SELECTED)
NUM_MAG_CHANNELS = len(MAG_CHANNELS)

DATA_DIR = 'C:/data2_test3'


# -----------------------------------------------------------------------------
# Data Preprocessing and Loading
# -----------------------------------------------------------------------------
def create_valid_time_mask(time_array, available_intervals):
    mask = np.zeros_like(time_array, dtype=bool)
    if available_intervals.ndim == 1: available_intervals = available_intervals.reshape(1, -1)
    for t_start, t_end in available_intervals:
        start_idx, end_idx = np.searchsorted(time_array, [t_start, t_end])
        mask[start_idx:end_idx] = True
    return mask


def create_and_fit_scalers(file_list):
    print("Fitting independent scalers...")
    ece_data, mag_data, freq_data, strength_data = [], [], [], []
    for path in tqdm(file_list, desc="Collecting scaler data"):
        try:
            with h5py.File(path, 'r') as f:
                if 'available_time' not in f or 'tearing_mode_time' not in f: continue
                time, label_time = f['time'][:], f['tearing_mode_time'][:]
                valid_time_mask = create_valid_time_mask(time, f['available_time'][:])
                freq_interp = np.interp(time, label_time, f['tearing_mode_freq'][:])
                strength_interp = np.interp(time, label_time, f['tearing_mode_a'][:])
                final_valid_indices = np.where(valid_time_mask & (freq_interp > 0))[0]

                if len(final_valid_indices) > 0:
                    for ch in ECE_CHANNELS_SELECTED: ece_data.append(f[ch][:][final_valid_indices])
                    for ch in MAG_CHANNELS: mag_data.append(f[ch][:][final_valid_indices])
                    freq_data.append(freq_interp[final_valid_indices])
                    strength_data.append(strength_interp[final_valid_indices])
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not freq_data: raise ValueError("No valid data found for scalers.")

    ECE_scaler = StandardScaler().fit(np.concatenate(ece_data).reshape(-1, 1))
    MAG_scaler = StandardScaler().fit(np.concatenate(mag_data).reshape(-1, 1))
    scaler_freq = StandardScaler().fit(np.concatenate(freq_data).reshape(-1, 1))
    scaler_strength = StandardScaler().fit(np.concatenate(strength_data).reshape(-1, 1))

    return ECE_scaler, MAG_scaler, scaler_freq, scaler_strength


class TearingDataset(Dataset):
    def __init__(self, X, y_cls, y_freq, y_strength, mag_indices):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        self.y_freq = torch.tensor(y_freq, dtype=torch.float32)
        self.y_strength = torch.tensor(y_strength, dtype=torch.float32)
        self.mag_indices = mag_indices
        self.nperseg, self.noverlap = 256, 128

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        mag_signals = self.X[idx][self.mag_indices]
        stft_list = [np.abs(signal.stft(ch.numpy(), fs=INPUT_SAMPLING_RATE,
                                        nperseg=self.nperseg, noverlap=self.noverlap)[2]) for ch in mag_signals]
        return x, torch.tensor(np.stack(stft_list, axis=0), dtype=torch.float32), self.y_cls[idx], self.y_freq[idx], \
        self.y_strength[idx]


def build_final_dataset(file_list, ECE_scaler, MAG_scaler, scaler_freq, scaler_strength, max_samples, name="Dataset"):
    pos_samples, neg_samples = [], []
    for path in tqdm(file_list, desc=f"Building {name}"):
        try:
            with h5py.File(path, 'r') as f:
                if 'available_time' not in f or not all(ch in f for ch in ALL_CHANNELS): continue
                time, label_time = f['time'][:], f['tearing_mode_time'][:]
                valid_time_mask = create_valid_time_mask(time, f['available_time'][:])

                signals_scaled = np.zeros((len(ALL_CHANNELS), len(time)))
                for i, ch_name in enumerate(ALL_CHANNELS):
                    scaler = ECE_scaler if ch_name in ECE_CHANNELS_SELECTED else MAG_scaler
                    signals_scaled[i, :] = scaler.transform(f[ch_name][:].reshape(-1, 1)).flatten()

                freq_interp = np.interp(time, label_time, f['tearing_mode_freq'][:])
                strength_interp = np.interp(time, label_time, f['tearing_mode_a'][:])

                for start in range(0, len(time) - WINDOW_SIZE, STEP_SIZE):
                    if not valid_time_mask[start + WINDOW_SIZE - 1]: continue
                    window = signals_scaled[:, start:start + WINDOW_SIZE]
                    freq_label = freq_interp[start + WINDOW_SIZE - 1]

                    if np.isnan(freq_label) or np.isnan(window).any(): continue

                    if freq_label > 0:
                        pos_samples.append((window, 1, freq_label, strength_interp[start + WINDOW_SIZE - 1]))
                    else:
                        neg_samples.append((window, 0, freq_label, strength_interp[start + WINDOW_SIZE - 1]))
        except Exception:
            pass

    np.random.shuffle(pos_samples)
    np.random.shuffle(neg_samples)

    pos_n = min(len(pos_samples), max_samples // 2)
    neg_n = min(len(neg_samples), max_samples - pos_n)
    final_samples = pos_samples[:pos_n] + neg_samples[:neg_n]

    if not final_samples: return None

    X, y_cls, freq, strength = zip(*final_samples)
    y_freq_scaled = scaler_freq.transform(np.array(freq).reshape(-1, 1))
    y_strength_scaled = scaler_strength.transform(np.array(strength).reshape(-1, 1))

    return TearingDataset(np.array(X), np.array(y_cls), y_freq_scaled, y_strength_scaled, mag_indices)


# File splitting logic
all_file_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.h5')]
test_shot_numbers = {'86486', '95753', '112636', '112903'}
val_shot_numbers = {'86487', '91413', '91706', '95926', '98192', '100037'}

train_val_files, test_files = [], []
for path in all_file_paths:
    shot = ''.join(filter(str.isdigit, os.path.splitext(os.path.basename(path))[0]))
    (test_files if shot in test_shot_numbers else train_val_files).append(path)

new_val_files = [p for p in train_val_files if
                 ''.join(filter(str.isdigit, os.path.splitext(os.path.basename(p))[0])) in val_shot_numbers]
new_train_files = [p for p in train_val_files if
                   ''.join(filter(str.isdigit, os.path.splitext(os.path.basename(p))[0])) not in val_shot_numbers]

print(f"Data Split: {len(new_train_files)} Train, {len(new_val_files)} Val, {len(test_files)} Test")

if new_train_files:
    ECE_scaler, MAG_scaler, scaler_freq, scaler_strength = create_and_fit_scalers(all_file_paths)
    train_dataset = build_final_dataset(new_train_files, ECE_scaler, MAG_scaler, scaler_freq, scaler_strength,
                                        MAX_TRAIN_SAMPLES, "Train")
    val_dataset = build_final_dataset(new_val_files, ECE_scaler, MAG_scaler, scaler_freq, scaler_strength,
                                      int(MAX_TRAIN_SAMPLES * 0.2), "Val")
    test_dataset = build_final_dataset(test_files, ECE_scaler, MAG_scaler, scaler_freq, scaler_strength, 8000, "Test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             drop_last=True) if test_dataset else None
else:
    raise ValueError("No training files found.")


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------
class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size) if pool_size else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=(2, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size) if pool_size else nn.Identity()

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class MagTimeDomainExtractor(nn.Module):
    def __init__(self, num_mag_channels, d_model=256, feature_dims=[32, 64, 128]):
        super().__init__()
        layers = []
        in_channels = num_mag_channels
        for out_channels in feature_dims:
            layers.append(Conv1DBlock(in_channels, out_channels))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feature_dims[-1], d_model)

    def forward(self, x):
        x = self.network(x)
        x = self.final_pool(x).squeeze(-1)
        return self.fc(x)


class MagFreqDomainExtractor(nn.Module):
    def __init__(self, num_mag_channels, freq_bins, d_model=256, feature_dims=[32, 64, 128]):
        super().__init__()
        layers = []
        in_channels = num_mag_channels
        for out_channels in feature_dims:
            layers.append(Conv2DBlock(in_channels, out_channels, pool_size=(2, 1)))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dims[-1], d_model)

    def forward(self, x):
        x = self.network(x)
        x = self.final_pool(x).view(x.size(0), -1)
        return self.fc(x)


class SpatiotemporalECEExtractor(nn.Module):
    def __init__(self, num_ece_channels, window_size, d_model=256, feature_dims=[32, 64]):
        super().__init__()
        time_layers = []
        in_channels = num_ece_channels
        for out_channels in feature_dims:
            time_layers.append(Conv1DBlock(in_channels, out_channels, pool_size=2))
            in_channels = out_channels
        self.temporal_network = nn.Sequential(*time_layers)

        final_feature_dim = feature_dims[-1]
        self.spatial_fc = nn.Linear(num_ece_channels, final_feature_dim)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(final_feature_dim * 2, d_model)

    def forward(self, x):
        f_temp = self.temporal_network(x)
        f_temp_pooled = self.final_pool(f_temp).squeeze(-1)
        f_spat = F.relu(self.spatial_fc(x.permute(0, 2, 1)))
        f_spat_pooled = self.final_pool(f_spat.permute(0, 2, 1)).squeeze(-1)
        f_combined = torch.cat([f_temp_pooled, f_spat_pooled], dim=1)
        return self.fc(f_combined)


def info_nce_loss(anchor, positive, negatives, temperature=0.1):
    eps = 1e-6
    anchor = F.normalize(anchor + eps, p=2, dim=1)
    positive = F.normalize(positive + eps, p=2, dim=1)
    negatives = F.normalize(negatives + eps, p=2, dim=1)
    l_pos = torch.bmm(anchor.unsqueeze(1), positive.unsqueeze(2)).squeeze(-1)
    l_neg = torch.mm(anchor, negatives.t())
    logits = torch.cat([l_pos, l_neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits / temperature, labels)


class MTCFNet(nn.Module):
    def __init__(self, window_size, num_ece_channels, num_mag_channels, freq_bins, d_model=256, lstm_hidden_size=256):
        super().__init__()
        self.mag_time_extractor = MagTimeDomainExtractor(num_mag_channels, d_model=d_model)
        self.mag_freq_extractor = MagFreqDomainExtractor(num_mag_channels, freq_bins, d_model=d_model)
        self.ece_time_extractor = SpatiotemporalECEExtractor(num_ece_channels, window_size, d_model=d_model)

        # Input dimension: d_model * 3 (Time + Freq + ECE)
        self.bilstm = nn.LSTM(d_model * 3, lstm_hidden_size, batch_first=True, bidirectional=True)

        lstm_out_dim = lstm_hidden_size * 2
        self.head = nn.Sequential(nn.Linear(lstm_out_dim, lstm_out_dim // 2), nn.ReLU(), nn.Dropout(0.5))

        self.classifier = nn.Linear(lstm_out_dim // 2, 1)
        self.freq_regressor = nn.Linear(lstm_out_dim // 2, 1)
        self.strength_regressor = nn.Linear(lstm_out_dim // 2, 1)

    def forward(self, x_time, x_stft_mag):
        x_mag_time = x_time[:, mag_indices, :]
        x_ece_time = x_time[:, spatial_ece_indices, :]

        f_mag_time = self.mag_time_extractor(x_mag_time)
        f_mag_freq = self.mag_freq_extractor(x_stft_mag)
        f_ece_time = self.ece_time_extractor(x_ece_time)

        loss_contrastive = info_nce_loss(f_mag_time, f_mag_freq, f_ece_time) if self.training else torch.tensor(0.0)

        f_downstream = torch.cat([f_mag_time, f_mag_freq, f_ece_time], dim=1)

        context_vector = self.bilstm(f_downstream.unsqueeze(1))[0].squeeze(1)
        head_out = self.head(context_vector)

        cls_out = self.classifier(head_out)
        freq_out = self.freq_regressor(head_out)
        strength_out = self.strength_regressor(head_out)

        return cls_out, freq_out, strength_out, loss_contrastive


class AblationNet(MTCFNet):
    def __init__(self, window_size, num_ece_channels, num_mag_channels, freq_bins, d_model=256, lstm_hidden_size=256):
        super().__init__(window_size, num_ece_channels, num_mag_channels, freq_bins, d_model, lstm_hidden_size)

    def forward(self, x_time, x_stft_mag):
        x_mag_time = x_time[:, mag_indices, :]
        x_ece_time = x_time[:, spatial_ece_indices, :]

        f_mag_time = self.mag_time_extractor(x_mag_time)
        f_mag_freq = self.mag_freq_extractor(x_stft_mag)
        f_ece_time = self.ece_time_extractor(x_ece_time)

        # Force contrastive loss to zero
        loss_contrastive = torch.tensor(0.0, device=x_time.device)

        f_downstream = torch.cat([f_mag_time, f_mag_freq, f_ece_time], dim=1)

        context_vector = self.bilstm(f_downstream.unsqueeze(1))[0].squeeze(1)
        head_out = self.head(context_vector)

        cls_out = self.classifier(head_out)
        freq_out = self.freq_regressor(head_out)
        strength_out = self.strength_regressor(head_out)

        return cls_out, freq_out, strength_out, loss_contrastive


# -----------------------------------------------------------------------------
# Training Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Train Main MTCF-Net
    print("\n[1/2] Training MTCF-Net (Main Model)...")
    model = MTCFNet(WINDOW_SIZE, NUM_SPATIAL_ECE_CHANNELS, NUM_MAG_CHANNELS, freq_bins=129).to(DEVICE)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

    MAIN_MODEL_PATH = 'best_mag_ece_contrastive_model.pt'
    W_CONTRASTIVE = 1.0
    best_val_loss = float('inf')

    if train_loader:
        for epoch in range(EPOCHS):
            model.train()
            for x_time, x_stft_mag, y_cls, y_freq, y_strength in tqdm(train_loader, desc=f"MTCF-Net Epoch {epoch + 1}"):
                x_time, x_stft_mag = x_time.to(DEVICE), x_stft_mag.to(DEVICE)
                y_cls, y_freq, y_strength = y_cls.to(DEVICE), y_freq.to(DEVICE), y_strength.to(DEVICE)

                cls_out, freq_out, strength_out, loss_contrastive = model(x_time, x_stft_mag)

                loss_supervised = criterion_cls(cls_out.squeeze(-1), y_cls)
                if (mask := y_cls == 1).any():
                    loss_supervised += criterion_reg(freq_out[mask], y_freq[mask]) + \
                                       criterion_reg(strength_out[mask], y_strength[mask])

                loss = loss_supervised + W_CONTRASTIVE * loss_contrastive
                optimizer.zero_grad();
                loss.backward();
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_time, x_stft_mag, y_cls, y_freq, y_strength in val_loader:
                    x_time, x_stft_mag = x_time.to(DEVICE), x_stft_mag.to(DEVICE)
                    y_cls, y_freq, y_strength = y_cls.to(DEVICE), y_freq.to(DEVICE), y_strength.to(DEVICE)

                    cls_out, freq_out, strength_out, _ = model(x_time, x_stft_mag)
                    loss = criterion_cls(cls_out.squeeze(-1), y_cls)
                    if (mask := y_cls == 1).any():
                        loss += criterion_reg(freq_out[mask], y_freq[mask]) + \
                                criterion_reg(strength_out[mask], y_strength[mask])
                    val_loss += loss.item() * x_time.size(0)

            val_loss /= len(val_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MAIN_MODEL_PATH)
            scheduler.step()

    # 2. Train AblationNet
    print("\n[2/2] Training AblationNet (No Contrastive Loss)...")
    ablation_model = AblationNet(WINDOW_SIZE, NUM_SPATIAL_ECE_CHANNELS, NUM_MAG_CHANNELS, freq_bins=129).to(DEVICE)
    ablation_optimizer = optim.Adam(ablation_model.parameters(), lr=1e-4)
    ABLATION_MODEL_PATH = 'ablation_model.pt'

    if train_loader:
        for epoch in range(EPOCHS):
            ablation_model.train()
            for x_time, x_stft_mag, y_cls, y_freq, y_strength in tqdm(train_loader,
                                                                      desc=f"AblationNet Epoch {epoch + 1}"):
                x_time, x_stft_mag = x_time.to(DEVICE), x_stft_mag.to(DEVICE)
                y_cls, y_freq, y_strength = y_cls.to(DEVICE), y_freq.to(DEVICE), y_strength.to(DEVICE)

                cls_out, freq_out, strength_out, _ = ablation_model(x_time, x_stft_mag)

                loss = criterion_cls(cls_out.squeeze(-1), y_cls)
                if (mask := y_cls == 1).any():
                    loss += criterion_reg(freq_out[mask], y_freq[mask]) + \
                            criterion_reg(strength_out[mask], y_strength[mask])

                ablation_optimizer.zero_grad();
                loss.backward();
                ablation_optimizer.step()

        torch.save(ablation_model.state_dict(), ABLATION_MODEL_PATH)

    print("All training tasks completed.")