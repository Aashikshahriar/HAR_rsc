"""
evaluation.py

Evaluate trained CNN + Temporal Attention HAR model.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

from data_preprocessing import load_dataset, split_dataset, create_dataloaders
from model import build_model


def main():

    # =========================================
    # Paths
    # =========================================
    base_path = r"D:\ CSI-HAR-Dataset\CSI-HAR-Dataset"
    model_path = "best_cnn_attention.pth"

    print("Dataset path:", base_path)
    print("Model path:", model_path)

    # =========================================
    # Device
    # =========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================
    # Load Dataset
    # =========================================
    X, y, le, num_cols = load_dataset(base_path)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    _, val_loader = create_dataloaders(
        X_train,
        X_test,
        y_train,
        y_test,
        batch_size=64
    )

    # =========================================
    # Load Model
    # =========================================
    model = build_model(num_cols, len(np.unique(y)), device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("✅ Model loaded successfully")

    # =========================================
    # Evaluation
    # =========================================
    preds = []
    labs = []

    with torch.no_grad():
        for xb, yb in val_loader:

            xb = xb.to(device)

            outputs = model(xb)

            preds.extend(outputs.argmax(1).cpu().numpy())
            labs.extend(yb.numpy())

    # =========================================
    # Metrics
    # =========================================
    acc = accuracy_score(labs, preds)
    print(f"\n🎯 Test Accuracy: {acc:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(labs, preds, target_names=le.classes_))

    # =========================================
    # Confusion Matrix
    # =========================================
    cm = confusion_matrix(labs, preds)

    print("\n🧩 Confusion Matrix:")
    print(cm)

"""
======================== EVALUATION OPTIONS ========================

This script currently evaluates:
- Accuracy
- Classification Report
- Confusion Matrix

Users can extend evaluation with the following:

1. Per-Class Accuracy
   → Helps identify which activities perform poorly
   → Compute accuracy for each class separately

2. Precision, Recall, F1-score (macro / weighted)
   → Already included in classification_report
   → Can log separately for research comparison

3. Confusion Matrix Visualization
   → Plot using matplotlib/seaborn for better interpretation

4. ROC Curve / AUC (for multi-class)
   → Requires probability outputs (use softmax)
   → Useful for deeper performance analysis

5. Top-K Accuracy
   → Useful if multiple predictions are acceptable (e.g., top-3)

6. Inference Time / Latency
   → Measure model speed (important for real-time HAR systems)

7. Model Size & Parameters
   → Helps evaluate deployment feasibility

8. Robustness Testing
   → Add noise to input and test stability

9. Cross-Validation
   → Instead of single train-test split for more reliable results

10. Class Imbalance Metrics
   → Use weighted metrics if dataset is imbalanced

--------------------------------------------------------------------

Example snippets:

# Softmax probabilities (for ROC or confidence)
probs = torch.softmax(outputs, dim=1)

# Per-class accuracy
for i in range(len(le.classes_)):
    idx = [j for j, label in enumerate(labs) if label == i]
    class_acc = accuracy_score(
        [labs[j] for j in idx],
        [preds[j] for j in idx]
    )
    print(f"{le.classes_[i]} accuracy: {class_acc:.4f}")

# Inference time
import time
start = time.time()
_ = model(xb)
end = time.time()
print("Inference time:", end - start)

====================================================================
"""




# =========================================
# ENTRY POINT
# =========================================
if __name__ == "__main__":
    main()
