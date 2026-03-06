"""
acc_vs_training_samples.py

Generic experiment script to measure model accuracy vs number of training samples.

Works with any dataset and any PyTorch model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# ==========================================
# Utility: epsilon from sample size
# ==========================================
def compute_epsilon(N, c=5.0):
    """
    Noise scale based on training sample size.
    """
    return c / np.sqrt(N)


# ==========================================
# Main Experiment
# ==========================================
def run_sample_efficiency_experiment(
        X,
        y,
        model_class,
        input_dim,
        num_classes,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_fractions=(0.2, 0.4, 0.6, 0.8, 1.0),
        epochs=20,
        batch_size=64
):

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    train_sizes = []
    accuracies = []

    # -----------------------------
    # Loop over dataset fractions
    # -----------------------------
    for frac in train_fractions:

        if frac < 1.0:
            X_tr, _, y_tr, _ = train_test_split(
                X_train,
                y_train,
                train_size=frac,
                stratify=y_train,
                random_state=42
            )
        else:
            X_tr, y_tr = X_train, y_train

        N = len(X_tr)
        epsilon = compute_epsilon(N)

        # Noise augmentation
        X_tr = X_tr + np.random.normal(0, epsilon, X_tr.shape)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long)
            ),
            batch_size=batch_size,
            shuffle=True
        )

        # -------------------------
        # Model
        # -------------------------
        model = model_class(input_dim, num_classes).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_acc = 0.0

        # -------------------------
        # Training
        # -------------------------
        for _ in range(epochs):

            model.train()

            for xb, yb in train_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()

                loss = criterion(model(xb), yb)

                loss.backward()

                optimizer.step()

            # ---------------------
            # Validation
            # ---------------------
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

            best_acc = max(best_acc, acc)

        train_sizes.append(N)
        accuracies.append(best_acc)

        print(f"Samples: {N} | Accuracy: {best_acc:.4f}")

    return train_sizes, accuracies


# ==========================================
# Plot Function
# ==========================================
def plot_sample_efficiency(train_sizes, accuracies):

    plt.figure(figsize=(6, 4))

    plt.plot(
        train_sizes,
        np.array(accuracies) * 100,
        "o-",
        linewidth=2
    )

    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy (%)")

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    plt.show()



# ==========================================
# Example Usage
# ==========================================

# from acc_vs_training_samples import run_sample_efficiency_experiment, plot_sample_efficiency
# from model import CNNAttention
#
# # X -> dataset features
# # y -> labels
#
# train_sizes, accuracies = run_sample_efficiency_experiment(
#     X,
#     y,
#     model_class=CNNAttention,
#     input_dim=X.shape[-1],
#     num_classes=len(set(y))
# )
#
# plot_sample_efficiency(train_sizes, accuracies)