"""
dp_experiment.py

Differential Privacy experiment using Opacus.

Runs epsilon sweep experiments to evaluate
accuracy vs privacy budget (ε).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


# ==========================================
# Differential Privacy Training
# ==========================================
def train_with_dp(
    model_class,
    train_dataset,
    test_dataset,
    input_dim,
    num_classes,
    epsilon,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=30,
    batch_size=64,
):

    # -----------------------------
    # Model
    # -----------------------------
    model = model_class(input_dim, num_classes).to(device)

    # Fix layers for DP compatibility
    model = ModuleValidator.fix(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Privacy Engine
    # -----------------------------
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=1e-5,
        epochs=epochs, #update epoch number for longer training
        max_grad_norm=1.0
    )

    # -----------------------------
    # Training
    # -----------------------------
    for _ in range(epochs):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            loss = criterion(model(xb), yb)

            loss.backward()

            optimizer.step()

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()

    preds = []
    labs = []

    with torch.no_grad():

        for xb, yb in test_loader:

            xb = xb.to(device)

            outputs = model(xb)

            preds.extend(outputs.argmax(1).cpu().numpy())

            labs.extend(yb.numpy())

    acc = accuracy_score(labs, preds)

    return acc


# ==========================================
# Epsilon Sweep Experiment
# ==========================================
def run_dp_experiment(
    model_class,
    train_dataset,
    test_dataset,
    input_dim,
    num_classes,
    epsilons=(2, 4, 6, 8, 10, 12, 14), #update epsilon values for more privacy budgets
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    results = {}

    for eps in epsilons:

        acc = train_with_dp(
            model_class,
            train_dataset,
            test_dataset,
            input_dim,
            num_classes,
            epsilon=eps,
            device=device
        )

        results[eps] = acc

        print(f"ε = {eps:>2} | Accuracy = {acc:.4f}")

    print("\n🔐 Differential Privacy Results")

    for k, v in results.items():

        print(f"Epsilon {k:>2}: Accuracy {v:.4f}")

    return results


# ==========================================
# Example Usage
# ==========================================

# from dp_experiment import run_dp_experiment
# from model import CNNAttention
#
# results = run_dp_experiment(
#     model_class=CNNAttention,
#     train_dataset=train_ds,
#     test_dataset=test_ds,
#     input_dim=num_cols,
#     num_classes=len(np.unique(y))
# )
