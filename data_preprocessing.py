"""
data_preprocessing.py

Handles loading, preprocessing, and dataloaders
for CSI HAR dataset.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# =========================================
# Helper Functions
# =========================================
def window_slicing(sample, window_size=128, stride=32): #window size and stride can be tuned
    slices = []
    for start in range(0, sample.shape[0] - window_size + 1, stride):
        slices.append(sample[start:start + window_size, :])
    return slices


def add_gaussian_noise(data, std=0.03):
    return data + np.random.normal(0.0, std, data.shape)  # noise can be tuned


# =========================================
# Load Dataset
# =========================================
def load_dataset(base_path):

    activities = ["bend", "fall", "lie down", "run", "sitdown", "standup", "walk"] #change this if your dataset has different activities

    min_rows, num_cols = None, None
    data_list, labels_list = [], []

    # Find minimum length
    for act in activities:
        files = [f for f in os.listdir(os.path.join(base_path, act)) if f.endswith("_A.csv")] #change this if your files have different naming pattern
        for f in files:
            df = pd.read_csv(os.path.join(base_path, act, f), header=None)
            min_rows = df.shape[0] if min_rows is None else min(min_rows, df.shape[0])
            num_cols = df.shape[1]

    # Load + Normalize + Augment
    for act in activities:
        files = [f for f in os.listdir(os.path.join(base_path, act)) if f.endswith("_A.csv")] #change this if your files have different naming pattern
        for f in files:
            df = pd.read_csv(os.path.join(base_path, act, f), header=None)

            df = df.iloc[:min_rows]

            # Normalize
            df = (df - df.mean()) / (df.std() + 1e-6)

            # Window slicing + augmentation
            for slice_ in window_slicing(df.values):
                data_list.append(slice_)
                labels_list.append(act)

                # Augmentation
                data_list.append(add_gaussian_noise(slice_))
                labels_list.append(act)

    X = np.stack(data_list)
    y = np.array(labels_list)

    print("Dataset shape:", X.shape)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le, num_cols


# =========================================
# Train-Test Split
# =========================================
def split_dataset(X, y):

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


# =========================================
# Create DataLoaders
# =========================================
def create_dataloaders(X_train, X_test, y_train, y_test,
                       batch_size=64):

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader
