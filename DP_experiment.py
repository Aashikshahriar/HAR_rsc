# install opacus if not already installed by !pip install opacus
"""
train_dp.py

Training CNN + Temporal Attention with Differential Privacy (Opacus)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from data_preprocessing import load_dataset, split_dataset, create_dataloaders
from model import build_model

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def main():

    # =========================================
    # Path
    # =========================================
    base_path = r"D:\ CSI-HAR-Dataset\CSI-HAR-Dataset"

    # =========================================
    # Device
    # =========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # =========================================
    # Load Data
    # =========================================
    X, y, le, num_cols = load_dataset(base_path)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    train_loader, val_loader = create_dataloaders(
        X_train, X_test, y_train, y_test,
        batch_size=64
    )

    # =========================================
    # Model
    # =========================================
    model = build_model(num_cols, len(np.unique(y)), device)

    # 🔥 REQUIRED for Opacus
    model = ModuleValidator.fix(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # =========================================
    # Differential Privacy Setup
    # =========================================
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=12,        # 🔥 adjust privacy level
        target_delta=1e-5,
        epochs=60,
        max_grad_norm=1.0          # 🔥 adjust gradient clipping
    )

    print("🔐 Differential Privacy Enabled")

    # =========================================
    # Training Loop
    # =========================================
    best_acc = 0

    for epoch in range(60):   #🔥 adjust epoch number for longer training

        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)

            loss.backward()
            optimizer.step()

        # ==========================
        # Validation
        # ==========================
        model.eval()
        preds, labs = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)

                preds.extend(out.argmax(1).cpu().numpy())
                labs.extend(yb.numpy())

        acc = accuracy_score(labs, preds)

        # 🔐 Get privacy budget spent
        epsilon = privacy_engine.get_epsilon(delta=1e-5)

        print(f"Epoch {epoch+1:02d} | Acc: {acc:.4f} | ε: {epsilon:.2f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_dp_model.pth")

    print(f"\n✅ Best DP Accuracy: {best_acc:.4f}")


# =========================================
# ENTRY POINT
# =========================================
if __name__ == "__main__":
    main()






"""
==================== EXTENSIONS & USER OPTIONS ====================

This script implements CNN + Temporal Attention with Differential Privacy (Opacus).

Users can extend or modify this implementation in several ways:

---------------------------
🔐 Differential Privacy
---------------------------
- Tune epsilon (privacy level):
    Lower epsilon → stronger privacy, lower accuracy
    Higher epsilon → weaker privacy, better accuracy

- Adjust max_grad_norm:
    Controls gradient clipping strength

- Modify delta:
    Typically set as 1 / dataset size

- Perform epsilon sweep:
    Try multiple epsilon values (e.g., 2, 4, 6, 8, 10)
    and compare accuracy vs privacy trade-off

---------------------------
📊 Evaluation Improvements
---------------------------
- Add confusion matrix visualization
- Compute per-class accuracy
- Track precision, recall, F1-score separately
- Plot ROC curves (multi-class)
- Add Top-K accuracy evaluation

---------------------------
📈 Visualization
---------------------------
- Plot training/validation accuracy curves
- Plot loss curves
- Plot epsilon vs accuracy graph (privacy trade-off)

---------------------------
⚙️ Model Improvements
---------------------------
- Replace attention with Transformer encoder
- Add deeper CNN layers or residual connections
- Use multi-scale convolutions
- Experiment with different dropout rates

---------------------------
🧪 Data Improvements
---------------------------
- Apply more augmentation (noise, scaling, shifting)
- Balance dataset if classes are imbalanced
- Try different window sizes and strides

---------------------------
🚀 Performance Optimization
---------------------------
- Increase batch size (if memory allows)
- Use mixed precision training (FP16)
- Optimize DataLoader settings

---------------------------
📦 Deployment / Research
---------------------------
- Export model for inference (TorchScript / ONNX)
- Measure inference latency
- Compare DP vs non-DP model performance
- Use cross-validation for robust evaluation

------------------------------------------------------------------

This code serves as a strong baseline for:
- Privacy-preserving Human Activity Recognition (HAR)
- Research and experimentation with DP-SGD (Opacus)

==================================================================
"""
