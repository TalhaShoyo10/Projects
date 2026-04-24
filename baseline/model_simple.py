from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTCNN(nn.Module):
    """
    Minimal MNIST CNN.

    Input : (N, 1, 28, 28)
    Output: (N, 10) logits
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # (N,32,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (N,64,7,7)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_simple_model(weights_path: str | Path, device: torch.device | None = None) -> SimpleMNISTCNN:
    if device is None:
        device = default_device()
    weights_path = Path(weights_path)
    model = SimpleMNISTCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
