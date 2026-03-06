# =====================================================
# WiAR Height-Factor Preprocessing + Augmentation
# =====================================================
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- Config ----------------
BASE_DIR = "D:/WiAR-master/WiAR-master/height_factor_activity_data"
SAVE_BASE = "D:/WiAR-master/wiar_height_preprocessed"

HEIGHTS = ["60_data", "90_data", "120_data"]
HEIGHT_NAMES = {"60_data": "60cm", "90_data": "90cm", "120_data": "120cm"}

NUM_CLASSES = 16
SAMPLES_PER_CLASS = 30
TOTAL_SAMPLES = NUM_CLASSES * SAMPLES_PER_CLASS

GAUSSIAN_STD = 0.02 # tune std for more vairance
SCALE_RANGE = (0.95, 1.05)
SEED = 42

np.random.seed(SEED)
os.makedirs(SAVE_BASE, exist_ok=True)

# ---------------- Utils ----------------
def load_excel(path):
    return pd.read_excel(path, engine="xlrd").values

def add_gaussian_noise(x, std):
    return x + np.random.normal(0, std, x.shape)

def temporal_jitter(x):
    t = x.shape[0]
    idx = np.random.randint(0, t)
    if np.random.rand() > 0.5:
        x = np.delete(x, idx, axis=0)
        x = np.pad(x, ((0,1),(0,0),(0,0)), mode="edge")
    else:
        x = np.insert(x, idx, x[idx], axis=0)[:t]
    return x

def amplitude_scaling(x, scale_range):
    return x * np.random.uniform(*scale_range)

# =====================================================
# Main Loop (per height)
# =====================================================
for h in HEIGHTS:
    print(f"\n📥 Processing height: {HEIGHT_NAMES[h]}")

    data_dir = os.path.join(BASE_DIR, h)
    save_dir = os.path.join(SAVE_BASE, HEIGHT_NAMES[h])
    os.makedirs(save_dir, exist_ok=True)

    XA = load_excel(os.path.join(data_dir, "sample_antenna_A.xls"))
    XB = load_excel(os.path.join(data_dir, "sample_antenna_B.xls"))
    XC = load_excel(os.path.join(data_dir, "sample_antenna_C.xls"))

    min_len = min(len(XA), len(XB), len(XC))
    X_all = np.stack([XA[:min_len], XB[:min_len], XC[:min_len]], axis=1).astype(np.float32)

    T_sample = min_len // TOTAL_SAMPLES
    print("Time steps per sample:", T_sample)

    X_base, y_base = [], []
    idx = 0
    for cls in range(NUM_CLASSES):
        for _ in range(SAMPLES_PER_CLASS):
            seg = X_all[idx:idx+T_sample]
            if seg.shape[0] == T_sample:
                seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
                X_base.append(seg)
                y_base.append(cls)
            idx += T_sample

    X_base = np.array(X_base, dtype=np.float32)
    y_base = np.array(y_base, dtype=np.int64)

    # ---------------- Augmentation ----------------
    X_aug, y_aug = [], []
    for i in tqdm(range(len(X_base))):
        x, y = X_base[i], y_base[i]
        X_aug.append(x);                          y_aug.append(y)
        X_aug.append(add_gaussian_noise(x, GAUSSIAN_STD)); y_aug.append(y)
        X_aug.append(temporal_jitter(x));         y_aug.append(y)
        X_aug.append(amplitude_scaling(x, SCALE_RANGE)); y_aug.append(y)

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int64)

    np.save(os.path.join(save_dir, "X.npy"), X_aug)
    np.save(os.path.join(save_dir, "y.npy"), y_aug)

    meta = {
        "dataset": "WiAR",
        "factor": "height",
        "height": HEIGHT_NAMES[h],
        "num_classes": NUM_CLASSES,
        "original_samples": int(len(X_base)),
        "augmented_samples": int(len(X_aug)),
        "input_shape": list(X_aug.shape[1:])
    }

    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"✅ Saved preprocessed data for {HEIGHT_NAMES[h]}")
