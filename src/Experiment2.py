"""
experiment2.py
--------------
Full Experiment 2:
  - FocalLoss(alpha=1.0, gamma=2.0), 10 epochs
  - Saves checkpoint: checkpoint_focal.pth
  - Plots validation accuracy curve  → focal_val_accuracy.png
  - Generates confusion matrix on test set → focal_confusion_matrix.png

Run:
    python src/experiment2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")           # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__))
from loss import FocalLoss



class SimpleNet(nn.Module):
    def __init__(self, input_dim=20, num_classes=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)   # avg loss per batch


def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total



def plot_val_accuracy(val_accs, save_path="focal_val_accuracy.png"):
    epochs = list(range(1, len(val_accs) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_accs, marker="o", linewidth=2,
            color="#4C72B0", markerfacecolor="white", markeredgewidth=2, markersize=8)

    # annotate best epoch
    best_epoch = int(np.argmax(val_accs)) + 1
    best_acc   = max(val_accs)
    ax.annotate(f"Best: {best_acc:.3f} @ epoch {best_epoch}",
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + 0.4, best_acc - 0.015),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9, color="gray")

    ax.set_title("Experiment 2 — FocalLoss(α=1, γ=2)\nValidation Accuracy over 10 Epochs",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_ylim(0, max(val_accs) * 1.25)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved: {save_path}")


def plot_confusion_matrix(model, loader, device, num_classes,
                          save_path="focal_confusion_matrix.png"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(range(num_classes)))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")

    ax.set_title("Experiment 2 — FocalLoss\nConfusion Matrix on Test Set",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved: {save_path}")



if __name__ == "__main__":
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS      = 10
    LR          = 1e-3
    NUM_CLASSES = 15
    INPUT_DIM   = 20

    print(f"Using device: {device}\n")

torch.manual_seed(42)

    X_train = torch.randn(400, INPUT_DIM)
    y_train = torch.randint(0, NUM_CLASSES, (400,))
    dummy_train = DataLoader(TensorDataset(X_train, y_train),
                             batch_size=32, shuffle=True)

    X_val = torch.randn(100, INPUT_DIM)
    y_val = torch.randint(0, NUM_CLASSES, (100,))
    dummy_val = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    X_test = torch.randn(200, INPUT_DIM)
    y_test = torch.randint(0, NUM_CLASSES, (200,))
    dummy_test = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    
    model     = SimpleNet(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
    loss_fn   = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

       print("── Experiment 2: FocalLoss(alpha=1, gamma=2) — 10 Epochs ──\n")

    exp2_train_losses = []
    exp2_val_losses   = []
    exp2_val_accs     = []

    best_val_acc  = 0.0
    checkpoint_path = "checkpoint_focal.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss          = train_one_epoch(model, dummy_train, loss_fn, optimizer, device)
        val_loss, val_acc   = validate(model, dummy_val, loss_fn, device)

        exp2_train_losses.append(round(train_loss, 4))
        exp2_val_losses.append(round(val_loss, 4))
        exp2_val_accs.append(round(val_acc, 4))

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

              if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":          val_acc,
                "val_loss":         val_loss,
                "alpha":            1.0,
                "gamma":            2.0,
            }, checkpoint_path)
            print(f"           ✓ checkpoint saved (val_acc={val_acc:.4f})")

        print(f"\nexp2_train_losses = {exp2_train_losses}")
    print(f"exp2_val_losses   = {exp2_val_losses}")
    print(f"exp2_val_accs     = {exp2_val_accs}")
    print(f"\nBest val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

      print("\nGenerating plots...")
    plot_val_accuracy(exp2_val_accs,    save_path="focal_val_accuracy.png")
    plot_confusion_matrix(model, dummy_test, device, NUM_CLASSES,
                          save_path="focal_confusion_matrix.png")
