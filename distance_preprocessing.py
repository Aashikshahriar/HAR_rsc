# =====================================================
# WiAR Preprocessing with Augmentation , Just change /x_m data for 3 and 6m
# =====================================================
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================
# Configuration
# =====================================================
DATA_DIR = "D:/WiAR-master/WiAR-master/distance_factor_activity_data/1m_data"
SAVE_DIR = "D:/WiAR-master/wiar_1m_preprocessed_augmented"

NUM_CLASSES = 16
SAMPLES_PER_CLASS = 30
TOTAL_SAMPLES = NUM_CLASSES * SAMPLES_PER_CLASS

GAUSSIAN_STD = 0.02 # tune std for more vairance
SCALE_RANGE = (0.95, 1.05)
SEED = 42

os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(SEED)

# =====================================================
# Load Excel Files
# =====================================================
def load_excel(path):
    return pd.read_excel(path, engine="xlrd").values

print("📥 Loading WiAR raw CSI data...")

X_A = load_excel(os.path.join(DATA_DIR, "sample_antenna_A.xls"))
X_B = load_excel(os.path.join(DATA_DIR, "sample_antenna_B.xls"))
X_C = load_excel(os.path.join(DATA_DIR, "sample_antenna_C.xls"))

min_len = min(len(X_A), len(X_B), len(X_C))

# Shape: [T_total, 3, 200]
X_all = np.stack(
    [X_A[:min_len], X_B[:min_len], X_C[:min_len]],
    axis=1
).astype(np.float32)

print("Raw stacked shape:", X_all.shape)

# =====================================================
# Segment Continuous Stream into Samples
# =====================================================
T_total = X_all.shape[0]
T_sample = T_total // TOTAL_SAMPLES

print(f"Time steps per sample: {T_sample}")

X_samples, y_samples = [], []
idx = 0

for cls in range(NUM_CLASSES):
    for _ in range(SAMPLES_PER_CLASS):
        seg = X_all[idx:idx + T_sample]
        if seg.shape[0] == T_sample:
            # Per-sample normalization
            mean = np.mean(seg)
            std = np.std(seg) + 1e-6
            seg = (seg - mean) / std

            X_samples.append(seg)
            y_samples.append(cls)
        idx += T_sample

X_samples = np.array(X_samples, dtype=np.float32)  # [480, T, 3, 200]
y_samples = np.array(y_samples, dtype=np.int64)

print("Base dataset shape:", X_samples.shape)

# =====================================================
# Augmentation Functions (SAFE)
# =====================================================
def add_gaussian_noise(x, std=0.02):
    return x + np.random.normal(0, std, x.shape)

def temporal_jitter(x):
    """Duplicate or remove one frame (keeps length)"""
    t = x.shape[0]
    idx = np.random.randint(0, t)

    if np.random.rand() > 0.5:
        # Remove one frame
        x_new = np.delete(x, idx, axis=0)
        x_new = np.pad(x_new, ((0,1),(0,0),(0,0)), mode="edge")
    else:
        # Duplicate one frame
        x_new = np.insert(x, idx, x[idx], axis=0)
        x_new = x_new[:t]

    return x_new

def amplitude_scaling(x, scale_range=(0.95, 1.05)):
    scale = np.random.uniform(*scale_range)
    return x * scale

# =====================================================
# Apply Augmentation
# =====================================================
X_aug, y_aug = [], []

print("🧪 Applying augmentation...")

for i in tqdm(range(len(X_samples))):
    x = X_samples[i]
    y = y_samples[i]

    # Original
    X_aug.append(x)
    y_aug.append(y)

    # Gaussian noise
    X_aug.append(add_gaussian_noise(x, GAUSSIAN_STD))
    y_aug.append(y)

    # Temporal jitter
    X_aug.append(temporal_jitter(x))
    y_aug.append(y)

    # Amplitude scaling
    X_aug.append(amplitude_scaling(x, SCALE_RANGE))
    y_aug.append(y)

X_aug = np.array(X_aug, dtype=np.float32)
y_aug = np.array(y_aug, dtype=np.int64)

print("Augmented dataset shape:", X_aug.shape)

# =====================================================
# Final Save
# =====================================================
np.save(os.path.join(SAVE_DIR, "X.npy"), X_aug)
np.save(os.path.join(SAVE_DIR, "y.npy"), y_aug)

meta = {
    "dataset": "WiAR",
    "distance": "1m",
    "num_classes": NUM_CLASSES,
    "samples_per_class": SAMPLES_PER_CLASS,
    "original_samples": int(len(X_samples)),
    "augmented_samples": int(len(X_aug)),
    "input_shape": list(X_aug.shape[1:]),
    "augmentation": {
        "gaussian_noise_std": GAUSSIAN_STD,
        "temporal_jitter": True,
        "amplitude_scaling": SCALE_RANGE
    },
    "normalization": "per-sample z-score",
    "seed": SEED
}

with open(os.path.join(SAVE_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=4)

print("\n📦 Preprocessing complete!")
print(f"Saved to: {SAVE_DIR}")
print(json.dumps(meta, indent=2))
