import torch
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from src.model import build_model
from src.dataset import test_loader

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
model = build_model(num_classes=15).to(device)

# LOAD CHECKPOINT
model.load_state_dict(
    torch.load(
        "results/checkpoints/checkpoint_ce.pth",
        map_location=device
    )
)

print("Checkpoint loaded successfully!")

# EVALUATION MODE
model.eval()

all_preds = []
all_labels = []

# NO GRADIENTS NEEDED
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# CLASSIFICATION REPORT
print("\nEvaluation completed!\n")

print(
    classification_report(
        all_labels,
        all_preds
    )
)

# CONFUSION MATRIX
cm = confusion_matrix(all_labels, all_preds)

class_names = test_loader.dataset.classes

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.tight_layout()

plt.savefig("results/confusion_matrix.png")

print("\nConfusion matrix saved to results/confusion_matrix.png")

plt.show()

# SHOW 15 PREDICTION IMAGES
class_names = test_loader.dataset.classes

# ---------------- PREDICTION GRID ----------------

import matplotlib.pyplot as plt

class_names = test_loader.dataset.classes

shown_classes = set()

fig, axes = plt.subplots(3, 5, figsize=(15, 10))

axes = axes.flatten()

idx = 0

model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        for i in range(len(images)):

            true_label = labels[i].item()

            # skip if this class already shown
            if true_label in shown_classes:
                continue

            shown_classes.add(true_label)

            image = images[i].cpu().permute(1, 2, 0)

            axes[idx].imshow(image)

            true_name = class_names[true_label]
            pred_name = class_names[preds[i].item()]

            axes[idx].set_title(
                f"T: {true_name}\nP: {pred_name}",
                color="green" if true_label == preds[i].item() else "red",
                fontsize=10
            )

            axes[idx].axis("off")

            idx += 1

            # stop after 15 classes
            if idx == 15:
                break

        if idx == 15:
            break

plt.tight_layout()

plt.savefig("results/prediction_grid.png")

print("Prediction grid saved!")