import torch
import torch.nn as nn
import torch.nn.functional as F


def validate(model, loader, loss_fn, device):
    """
    Evaluate model on a validation DataLoader.

    Returns:
        avg_loss (float), accuracy (float in [0, 1])
    """
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss    = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy
