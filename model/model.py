import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    """
    Fully Connected Neural Network for MNIST digit classification.
    Architecture: 784 -> 256 -> 128 -> 10
    Kept for reference — CNN is preferred for this project.
    """
    def __init__(self):
        super().__init__()
        self.hidden_preactivation_1 = nn.Linear(784, 256)
        self.hidden_preactivation_2 = nn.Linear(256, 128)
        self.out_preactivation      = nn.Linear(128, 10)

    def forward(self, x):
        x          = x.view(x.size(0), -1)
        A_1        = F.relu(self.hidden_preactivation_1(x))
        A_2        = F.relu(self.hidden_preactivation_2(A_1))
        prediction = self.out_preactivation(A_2)
        return prediction


class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Simple baseline-style architecture:
        Conv2d(1 -> 32, 3x3, padding=1) → ReLU → MaxPool(2x2)   [32 x 14 x 14]
        Conv2d(32 -> 64, 3x3, padding=1) → ReLU → MaxPool(2x2)  [64 x 7  x 7 ]
        Flatten                                                   [3136]
        Linear(3136 -> 128) → ReLU
        Linear(128 -> 10)

    Input:  (batch, 1, 28, 28)  — grayscale MNIST images
    Output: (batch, 10)         — raw logits per class
    """
    def __init__(self):
        super().__init__()

        # ── Feature extraction ────────────────────────────────────────────
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Classification head ───────────────────────────────────────────
        # flatten size: 64 * 7 * 7 = 3136
        self.fc_1  = nn.Linear(64 * 7 * 7, 128)
        self.fc_2  = nn.Linear(128,  10)

    def forward(self, x):
        # Block 1: conv → relu → pool
        x = self.pool(F.relu(self.conv1(x)))     # (batch, 32, 13, 13)

        # Block 2: conv → relu → pool
        x = self.pool(F.relu(self.conv2(x)))     # (batch, 64, 5, 5)

        # Flatten everything except batch dimension
        x = x.flatten(start_dim=1)              # (batch, 1600)

        # Classification
        x          = F.relu(self.fc_1(x))       # (batch, 128)
        prediction = self.fc_2(x)               # (batch, 10)

        return prediction


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def load_model(weights_path: str, device: torch.device = None) -> CNN:
    """Load a saved CNN model from weights file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model
