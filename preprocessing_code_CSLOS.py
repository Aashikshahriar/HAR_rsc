import os
import glob
import re
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# =========================
# PATHS change environment for others, this is without noise augmented, so need to add noise. please refer to WiAR preprocessing code for noise augmentation
# =========================
env1_path = r"D:\v38wjmz6f6-1\Environment 1"
save_dir = r"D:\cslos_environment1_preprocessed"
os.makedirs(save_dir, exist_ok=True)

# =========================
# PARAMETERS
# =========================
TARGET_LEN = 900
WINDOW_SIZE = 300
STRIDE = 100
SEED = 42
np.random.seed(SEED)

# =========================
# ACTIVITY MAP
# =========================
activity_map = {
    1: 'Sit still on a chair',
    2: 'Falling down (sitting)',
    3: 'Lie down',
    4: 'Stand still',
    5: 'Falling down (standing)',
    6: 'Walking T→R',
    7: 'Turning',
    8: 'Walking R→T',
    9: 'Turning',
    10: 'Standing up',
    11: 'Sitting down',
    12: 'Pick a pen',
}

# =========================
# HELPERS
# =========================
def parse_filename(fname):
    match = re.match(r"E(\d+)_S(\d+)_C(\d+)_A(\d+)_T(\d+)\.csv", fname)
    return tuple(map(int, match.groups())) if match else None

def parse_csi_col(col):
    try:
        return complex(str(col).replace(' ', '').replace('i', 'j'))
    except:
        return np.nan

def window_slice(sample, win=300, stride=100):
    slices = []
    for start in range(0, sample.shape[0] - win + 1, stride):
        slices.append(sample[start:start+win])
    return slices

# =========================
# LOAD & PROCESS
# =========================
all_X, all_y, all_subjects = [], [], []

print("🔄 Preprocessing CSLOS Environment 1...")

subject_dirs = [os.path.join(env1_path, f"Subject {i}") for i in range(1, 11)]

for subj_dir in subject_dirs:
    csv_files = glob.glob(os.path.join(subj_dir, "*.csv"))
    for f in tqdm(csv_files):
        parsed = parse_filename(os.path.basename(f))
        if not parsed:
            continue

        _, subj, _, act, _ = parsed
        if act not in activity_map:
            continue

        df = pd.read_csv(f)
        csi_cols = [c for c in df.columns if c.startswith("csi_")]
        if len(csi_cols) != 90:
            continue

        csi = df[csi_cols].applymap(parse_csi_col).values
        amp = np.abs(csi)

        # Normalize
        mean, std = np.nanmean(amp), np.nanstd(amp)
        std = std if std > 1e-6 else 1.0
        amp = (amp - mean) / std
        amp = np.nan_to_num(amp)

        # Pad / truncate
        if amp.shape[0] < TARGET_LEN:
            pad = np.zeros((TARGET_LEN - amp.shape[0], 90))
            amp = np.vstack([amp, pad])
        else:
            amp = amp[:TARGET_LEN]

        # Window slicing (KEY PART)
        for w in window_slice(amp, WINDOW_SIZE, STRIDE):
            all_X.append(w.astype(np.float32))
            all_y.append(act)
            all_subjects.append(subj)

# =========================
# FINAL ARRAYS
# =========================
X = np.stack(all_X)        # [N, 300, 90]
y = np.array(all_y)
subjects = np.array(all_subjects)

print("✅ Final dataset shape:", X.shape)

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
y_enc = le.fit_transform(y)

# =========================
# SAVE
# =========================
np.save(os.path.join(save_dir, "X.npy"), X)
np.save(os.path.join(save_dir, "y.npy"), y_enc)
np.save(os.path.join(save_dir, "subjects.npy"), subjects)

with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

meta = {
    "target_len": TARGET_LEN,
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "num_samples": int(X.shape[0]),
    "num_classes": len(le.classes_),
    "activities": activity_map
}

with open(os.path.join(save_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=4)

print("📦 Preprocessed data saved to:", save_dir)
