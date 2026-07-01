import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset


def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=256, order=5):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = np.zeros_like(data)
    for i in range(data.shape[2]):
        filtered[:, :, i] = filtfilt(b, a, data[:, :, i], axis=0)
    return filtered


def normalize(data):
    for i in range(data.shape[2]):
        mean = data[:, :, i].mean(axis=0, keepdims=True)
        std  = data[:, :, i].std(axis=0, keepdims=True) + 1e-8
        data[:, :, i] = (data[:, :, i] - mean) / std
    return data


def select_channels(eeg_data, channel_names):
    """
    Select speech-relevant channels only.
    Falls back to all channels if none found.
    """
    indices = [i for i, ch in enumerate(channel_names)
               if ch in SPEECH_CHANNELS]
    if len(indices) < 8:
        print(f"  ⚠️  Only {len(indices)} speech channels found — using all 64")
        return eeg_data, list(range(eeg_data.shape[1]))
    print(f"  ✅ Selected {len(indices)} speech channels from {len(channel_names)}")
    return eeg_data[:, indices, :], indices


def sliding_window_augment(X, y, window=256, stride=16):
    """
    X: (trials, channels, time)
    Returns: (trials * n_windows, channels, window)
    """
    X_aug, y_aug = [], []
    T = X.shape[2]
    for i in range(len(X)):
        for start in range(0, T - window + 1, stride):
            X_aug.append(X[i, :, start:start + window])
            y_aug.append(y[i])
    return (np.array(X_aug, dtype=np.float32),
            np.array(y_aug,  dtype=np.int64))


def load_bci_data_cv(mat_file_train, mat_file_val,
                     include_validation=True):
    print(f"Loading train : {mat_file_train}")
    mat_train = sio.loadmat(mat_file_train)
    print(f"Loading val   : {mat_file_val}")
    mat_val   = sio.loadmat(mat_file_val)

    epo_train       = mat_train['epo_train']
    eeg_train       = epo_train['x'][0, 0]
    labels_oh_train = epo_train['y'][0, 0]
    fs              = int(epo_train['fs'][0, 0][0, 0])
    time_points     = epo_train['t'][0, 0].flatten()
    channel_names   = [ch[0] for ch in epo_train['clab'][0, 0].flatten()]
    class_names     = [cn[0] for cn in epo_train['className'][0, 0].flatten()]
    ax              = 0 if labels_oh_train.shape[0] == len(class_names) else 1
    labels_train    = np.argmax(labels_oh_train, axis=ax)

    if include_validation:
        epo_val       = mat_val['epo_validation']
        eeg_val       = epo_val['x'][0, 0]
        labels_oh_val = epo_val['y'][0, 0]
        labels_val    = np.argmax(labels_oh_val, axis=ax)
        eeg_data      = np.concatenate([eeg_train, eeg_val], axis=2)
        labels        = np.concatenate([labels_train, labels_val])
    else:
        eeg_data = eeg_train
        labels   = labels_train

    # Crop 0–2000ms
    # time_mask   = (time_points >= 0) & (time_points <= 2000)
    # eeg_data    = eeg_data[time_mask, :, :]
    # time_points = time_points[time_mask]

    # Bandpass + normalize
    # eeg_data = bandpass_filter(eeg_data, fs=fs)
    # eeg_data = normalize(eeg_data)

    # → (trials, channels, time)
    eeg_data = eeg_data.transpose(2, 1, 0).astype(np.float32)
    labels   = labels.astype(np.int64)

    # Select speech channels
    # eeg_data, ch_idx = select_channels(eeg_data, channel_names)
    # sel_channels     = [channel_names[i] for i in ch_idx]

    print(f"\n{'='*60}")
    print(f"  DATASET LOADED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  EEG shape    : {eeg_data.shape}  (trials × ch × time)")
    print(f"  Class dist   : {np.bincount(labels)}")
    print(f"  Sampling Hz  : {fs}")
    print(f"  Time window  : {time_points[0]:.1f} → {time_points[-1]:.1f} ms")
    print(f"  Channels     : {eeg_data.shape[1]}")
    print(f"  Classes      : {class_names}")
    print(f"{'='*60}\n")

    return eeg_data, labels, fs, time_points, class_names, channel_names


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


if __name__ == '__main__':
    DATA_PATH = '/media/csedept/cse2018/Project/Codes/2nd approach/C1/'
    X, y, fs, t, cls, channel_names = load_bci_data_cv(
        DATA_PATH + 'Train/Data_Sample01.mat',
        DATA_PATH + 'Validation/Data_Sample01.mat'
    )
    X_aug, y_aug = sliding_window_augment(X, y)
    print(f"Original  : {X.shape} → {len(y)} trials")
    print(f"Augmented : {X_aug.shape} → {len(y_aug)} windows")
    print("✅ dataloader.py OK!")
