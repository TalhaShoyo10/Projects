import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, roc_curve, classification_report
)

from model import CNN, load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_test_loader(batch_size: int = 64):
    test_split = datasets.MNIST(
        root="./data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ]
        ),
        download=True,
    )
    return DataLoader(test_split, batch_size=batch_size, shuffle=False)


def collect_predictions(model, loader):
    """Run inference and collect predictions, labels and probabilities."""
    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())

    return (torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy())


def print_metrics(preds, labels):
    print(f"Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"F1 (macro): {f1_score(labels, preds, average='macro'):.4f}")
    print(f"Recall   : {recall_score(labels, preds, average='macro'):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(labels, preds))


def plot_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png"); plt.show()


def plot_roc_curves(labels, probs):
    n = labels.shape[0]
    one_hot = np.zeros((n, 10))
    for i in range(n):
        one_hot[i, labels[i]] = 1

    plt.figure()
    for digit in range(10):
        fpr, tpr, _ = roc_curve(one_hot[:, digit], probs[:, digit])
        plt.plot(fpr, tpr, label=f"Digit {digit}")

    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend(loc="lower right")
    plt.savefig("roc_curves.png"); plt.show()


def visualise_incorrect(model, loader, limit: int = 5):
    """Display incorrectly classified samples."""
    shown = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu()
            for i, correct in enumerate(preds == labels):
                if not correct:
                    plt.imshow(images[i].squeeze(), cmap="gray")
                    plt.title(f"True: {labels[i]}  |  Predicted: {preds[i]}")
                    plt.axis("off"); plt.show()
                    shown += 1
                if shown >= limit:
                    return


def main():
    model = load_model("weights/best_model.bin", DEVICE)    
    loader = get_test_loader()
    preds, labels, probs = collect_predictions(model, loader)

    print_metrics(preds, labels)
    plot_confusion_matrix(preds, labels)
    plot_roc_curves(labels, probs)
    visualise_incorrect(model, loader, limit=2)


if __name__ == "__main__":
    main()
