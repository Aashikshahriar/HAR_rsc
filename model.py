"""
model.py

CNN + Temporal Attention model for CSI-based Human Activity Recognition.

Architecture:
    CNN Feature Extractor
        ↓
    Temporal Attention
        ↓
    Fully Connected Classifier
"""

import torch
import torch.nn as nn


# =========================================
# Temporal Attention Module
# =========================================
class TemporalAttention(nn.Module):
    """
    Temporal attention layer for sequence aggregation.

    Input:
        (batch_size, time_steps, feature_dim)

    Output:
        (batch_size, feature_dim)
    """

    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (B, T, C)

        weights = torch.softmax(self.attn(x), dim=1)

        # weighted sum
        context = torch.sum(weights * x, dim=1)

        return context


# =========================================
# CNN + Attention Model
# =========================================
class CNNAttention(nn.Module):
    """
    CNN + Temporal Attention model for HAR.
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()

        # ---------------------------
        # CNN Feature Extractor
        # ---------------------------
        self.cnn = nn.Sequential(

            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.GroupNorm(8, 64),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),

            nn.MaxPool1d(2),
            nn.Dropout(0.25)
        )

        # ---------------------------
        # Temporal Attention
        # ---------------------------
        self.attn = TemporalAttention(128)

        # ---------------------------
        # Classifier
        # ---------------------------
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # x shape: (B, T, C)

        x = x.permute(0, 2, 1)   # (B, C, T)

        x = self.cnn(x)

        x = x.permute(0, 2, 1)   # (B, T, C)

        x = self.attn(x)

        out = self.fc(x)

        return out


# =========================================
# Model Builder
# =========================================
def build_model(input_dim, num_classes, device):
    """
    Utility function to build the model.
    """

    model = CNNAttention(input_dim, num_classes)

    return model.to(device)

##### Example Usage #####

#from model import build_model

#model = build_model(num_cols, len(np.unique(y)), device)