from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from baseline.model_simple import SimpleMNISTCNN, default_device


ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "mnist_cnn.pt"

EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    device = default_device()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.MNIST(root=str(ROOT / "data"), train=True, transform=mnist_transform(), download=True)
    test_ds = datasets.MNIST(root=str(ROOT / "data"), train=False, transform=mnist_transform(), download=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
        running_loss = 0.0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            pbar.set_postfix(loss=running_loss / max(1, pbar.n))

        acc = evaluate(model, test_loader, device)
        print(f"Test accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"Saved: {WEIGHTS_PATH} (best_acc={best_acc:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()
