import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from src.model import build_model
from src.dataset import train_loader, val_loader
from src.loss import FocalLoss
from src.validate import validate


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """
    Runs one full pass over the training set

    Steps per batch:
    1. zero_grad - clear accumulated gradients
    2. forward - model predictions
    3. loss - compute loss
    4. backward - compute gradients
    5. step - update weights

    Returns:
        avg_loss (float)
    """

    model.train()
    running_loss = 0.0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    return avg_loss


if __name__ == "__main__":

    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # HYPERPARAMETERS
    EPOCHS = 10
    LR = 1e-4
    NUM_CLASSES = 15

    # =========================================================
    # EXPERIMENT 1 — CrossEntropyLoss
    # =========================================================

    print("\n── Experiment 1: CrossEntropyLoss ──")

    model = build_model(num_classes=NUM_CLASSES).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    exp1_train_losses = []

    # EARLY STOPPING VARIABLES
    best_val_acc = 0.0
    best_val_loss = float("inf")

    patience = 3
    counter = 0

    # TRAINING LOOP
    for epoch in range(1, EPOCHS + 1):

        # TRAIN
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device
        )

        # VALIDATE
        val_loss, val_acc = validate(
            model,
            val_loader,
            loss_fn,
            device
        )

        exp1_train_losses.append(round(train_loss, 4))

        # PRINT METRICS
        print(f"\nEpoch {epoch:02d}/{EPOCHS}")

        print(f"Train Loss: {train_loss:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")

        print(f"Validation Accuracy: {val_acc * 100:.2f}%")

        # SAVE BEST MODEL
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "results/checkpoints/checkpoint_ce.pth"
            )

            print(
                f"✓ checkpoint_ce.pth saved "
                f"(val_acc={val_acc * 100:.2f}%)"
            )

        # EARLY STOPPING
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            counter = 0

        else:

            counter += 1

            print(
                f"No improvement for "
                f"{counter} epoch(s)"
            )

        if counter >= patience:

            print("Early stopping triggered!")

            break

        # LEARNING RATE SCHEDULER
        scheduler.step()

        print("Scheduler step completed")

    print(f"\nexp1_train_losses = {exp1_train_losses}")

    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")

    # =========================================================
    # EXPERIMENT 2 — Focal Loss
    # =========================================================

    print("\n── Experiment 2: FocalLoss(alpha=1, gamma=2) ──")

    model2 = build_model(num_classes=NUM_CLASSES).to(device)

    loss_fn2 = FocalLoss(alpha=1.0, gamma=2.0)

    optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)

    scheduler2 = StepLR(
        optimizer2,
        step_size=5,
        gamma=0.1
    )

    exp2_train_losses = []
    exp2_val_losses = []
    exp2_val_accs = []

    best_val_acc2 = 0.0
    best_val_loss2 = float("inf")

    patience2 = 3
    counter2 = 0

    # TRAINING LOOP
    for epoch in range(1, EPOCHS + 1):

        # TRAIN
        train_loss = train_one_epoch(
            model2,
            train_loader,
            loss_fn2,
            optimizer2,
            device
        )

        # VALIDATE
        val_loss, val_acc = validate(
            model2,
            val_loader,
            loss_fn2,
            device
        )

        exp2_train_losses.append(round(train_loss, 4))
        exp2_val_losses.append(round(val_loss, 4))
        exp2_val_accs.append(round(val_acc, 4))

        # PRINT METRICS
        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        # SAVE BEST MODEL
        if val_acc > best_val_acc2:

            best_val_acc2 = val_acc

            torch.save(
                model2.state_dict(),
                "results/checkpoints/checkpoint_focal.pth"
            )

            print(
                f"✓ checkpoint_focal.pth saved "
                f"(val_acc={val_acc * 100:.2f}%)"
            )

        # EARLY STOPPING
        if val_loss < best_val_loss2:

            best_val_loss2 = val_loss
            counter2 = 0

        else:

            counter2 += 1

            print(
                f"No improvement for "
                f"{counter2} epoch(s)"
            )

        if counter2 >= patience2:

            print("Early stopping triggered!")

            break

        # SCHEDULER STEP
        scheduler2.step()

        print("Scheduler step completed")

    print(f"\nexp2_train_losses = {exp2_train_losses}")

    print(f"exp2_val_losses = {exp2_val_losses}")

    print(f"exp2_val_accs = {exp2_val_accs}")

    print(f"Best Validation Accuracy: {best_val_acc2 * 100:.2f}%")