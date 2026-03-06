"""
train.py

Training script for CNN + Temporal Attention HAR model.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from data_preprocessing import load_dataset, split_dataset, create_dataloaders #your preprocessing code
from model import build_model


# =========================================
# Device Setup
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

print(f"Using device: {device}")
print(f"GPUs available: {num_gpus}")


# =========================================
# Load Dataset
# =========================================
base_path = "/kaggle/input/csi-mdpi/CSI-HAR-Dataset" #change this path for others

X, y, le, num_cols = load_dataset(base_path)

X_train, X_test, y_train, y_test = split_dataset(X, y)

train_loader, val_loader = create_dataloaders(
    X_train,
    X_test,
    y_train,
    y_test
)


# =========================================
# Build Model
# =========================================
model = build_model(num_cols, len(np.unique(y)), device)

# Multi-GPU
if num_gpus > 1:
    model = nn.DataParallel(model)
    print("✅ DataParallel enabled")


# =========================================
# Training Setup
# =========================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10
)

best_acc = 0
patience = 12
counter = 0


# =========================================
# Training Loop
# =========================================
for epoch in range(60):   #update epoch number for longer training

    model.train()

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        outputs = model(xb)

        loss = criterion(outputs, yb)

        loss.backward()

        optimizer.step()

    scheduler.step()


    # ==========================
    # Validation
    # ==========================
    model.eval()

    preds = []
    labs = []

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)

            outputs = model(xb)

            preds.extend(outputs.argmax(1).cpu().numpy())
            labs.extend(yb.numpy())

    acc = accuracy_score(labs, preds)

    print(f"Epoch {epoch+1:02d} | Val Acc: {acc:.4f}")


    # ==========================
    # Early Stopping
    # ==========================
    if acc > best_acc:

        best_acc = acc

        torch.save(model.state_dict(), "best_cnn_attention.pth")

        counter = 0

    else:

        counter += 1

        if counter >= patience:

            print("⏹ Early stopping")

            break


print(f"\n✅ Best Validation Accuracy: {best_acc:.4f}")