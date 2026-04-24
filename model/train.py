import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import CNN, count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LR = 1e-3
BATCH_SIZE = 128
SAVE_PATH = "weights/best_model.bin"

SEED = 42
torch.manual_seed(SEED)

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# ── Canvas-style augmentation ─────────────────────────────────────────────────
#
# Thickens digit strokes during training to match what the drawing canvas
# produces after resize to 28x28. Without this the model only sees thin
# MNIST strokes and fails on canvas drawings.

# ── Transforms ───────────────────────────────────────────────────────────────
#
# Two separate pipelines:
#   train_transform  — augmented, applied only to training samples
#   eval_transform   — clean, applied to val and test
#
# Augmentations chosen carefully for MNIST:
#   RandomRotation(15)         slight tilt — digits are hand-written at angles
#   RandomAffine(translate)    small shifts — digit position varies in real use
#   RandomErasing(p=0.1)       randomly blacks out a small patch — mimics the
#                              noise / occlusion added by the CAPTCHA system
#   dilate_tensor              thickens strokes to match canvas drawing style
#
# Deliberately excluded:
#   Horizontal/Vertical flip   — flipping a 6 makes it a 9 (wrong label)
#   Large rotations            — 90° rotated 9 looks like a 6
#   Color jitter               — MNIST is grayscale, irrelevant

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


# ── Data ────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = BATCH_SIZE):
    # Download full datasets — we load them twice with different transforms
    # because torchvision MNIST applies one transform per dataset object.
    # Train indices get train_transform; val indices get eval_transform.
    full_train_aug   = datasets.MNIST(root="./data", train=True,
                                      transform=train_transform, download=True)
    full_train_clean = datasets.MNIST(root="./data", train=True,
                                      transform=eval_transform, download=False)
    test_dataset     = datasets.MNIST(root="./data", train=False,
                                      transform=eval_transform, download=True)

    # Get indices for train / val split
    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = random_split(range(len(full_train_aug)), [50000, 10000], generator=generator)

    # Augmented subset for training, clean subset for validation
    train_subset = Subset(full_train_aug,   list(train_indices))
    val_subset   = Subset(full_train_clean, list(val_indices))

    train_loader = DataLoader(train_subset,  batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset,    batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


# ── Train / Evaluate ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        preds = model(images)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (preds.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images)
            loss  = criterion(preds, labels)

            total_loss += loss.item()
            correct    += (preds.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curves"); plt.legend()
    plt.savefig("loss_curves.png"); plt.show()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs,   label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Accuracy Curves"); plt.legend()
    plt.savefig("accuracy_curves.png"); plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs("weights", exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders()

    model     = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Parameters: {count_parameters(model):,}")

    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → Best model saved (val acc: {best_val_acc:.4f})")

    plot_curves(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()
