import torch
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

print("\nConfusion Matrix:\n")
print(cm)

# SHOW 15 PREDICTION IMAGES
class_names = test_loader.dataset.classes

shown_classes = set()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        for i in range(len(images)):

            true_label = class_names[labels[i]]
            pred_label = class_names[preds[i]]

            if true_label not in shown_classes:

                img = images[i].cpu().permute(1, 2, 0)

                plt.imshow(img)

                plt.title(
                    f"True: {true_label} | Pred: {pred_label}"
                )

                plt.axis("off")

                plt.show()

                shown_classes.add(true_label)

            if len(shown_classes) == 15:
                break

        if len(shown_classes) == 15:
            break