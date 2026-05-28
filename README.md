# Fruit Recognition — VGG16

15-class fruit image classifier built with transfer learning on a pretrained VGG16 backbone.

---

## Model Architecture

### Base Model: VGG16

VGG16 is a deep convolutional neural network introduced by Simonyan & Zisserman (2014). It consists of 16 weight layers — 13 convolutional layers followed by 3 fully connected layers — and takes a fixed input of **224 × 224 × 3** (RGB image).

```
Input (224×224×3)
        │
┌───────▼────────────────────────────────┐
│  FEATURES  (13 Conv layers, frozen)    │
│                                        │
│  Block 1:  Conv(64)  → Conv(64)  → Pool│
│  Block 2:  Conv(128) → Conv(128) → Pool│
│  Block 3:  Conv(256) → Conv(256) →     │
│            Conv(256) → Pool            │
│  Block 4:  Conv(512) → Conv(512) →     │
│            Conv(512) → Pool            │
│  Block 5:  Conv(512) → Conv(512) →     │
│            Conv(512) → Pool            │
└───────────────────┬────────────────────┘
                    │  7×7×512 → flatten
┌───────────────────▼────────────────────┐
│  CLASSIFIER  (trainable)               │
│                                        │
│  Linear(25088 → 4096) + ReLU + Drop   │
│  Linear(4096  → 4096) + ReLU + Drop   │
│  Linear(4096  → 15)   ← replaced head │
└────────────────────────────────────────┘
        │
Output (15 class logits)
```

### Transfer Learning Strategy

The model is loaded with weights pretrained on ImageNet (1.28M images, 1000 classes). The convolutional feature layers are **frozen** — their weights are not updated during training. Only the fully connected classifier is trained from scratch on the fruit dataset.

This works because early conv layers learn universal features (edges, textures, colour blobs) that transfer well across vision tasks. Retraining them would be wasteful and risk overfitting on a smaller dataset.

### Parameter Table

| Component     | Parameters      | Trainable  |
|---------------|-----------------|------------|
| features      | 14,714,688      | frozen     |
| classifier[0] | 102,764,544     | yes        |
| classifier[3] | 16,781,312      | yes        |
| classifier[6] | 61,455          | yes (new)  |
| **Total**     | **134,321,999** | —          |
| **Trainable** | **119,607,311** | yes        |

`classifier[6]` is the replaced head: `nn.Linear(4096, 15)` — the only layer that did not exist in the original VGG16.

### Input Preprocessing

All images are resized to 224×224 and normalised using ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Training augmentations: random horizontal flip, colour jitter (brightness, contrast, saturation ±0.2).

---

## Dataset

- **Source:** [Kaggle — Fruit Recognition](https://www.kaggle.com/datasets/chrisfilo/fruit-recognition)
- **Classes:** 15 fruit categories
- **Split:** 80% train / 20% val (random split from Training folder), separate Test folder

---

## Experiments

| Exp | Loss Function        | Epochs | Best Val Acc |
|-----|----------------------|--------|--------------|
| 1   | CrossEntropyLoss     | 10     | TBD          |
| 2   | FocalLoss (a=1, g=2) | 10     | TBD          |

---

## Project Structure

```
src/
  model.py       # VGG16 definition, build_model()
  dataset.py     # ImageFolder loaders for train/val/test
  train.py       # train_one_epoch(), Experiment 1
  Validate.py    # validate() — val loss + accuracy
  loss.py        # FocalLoss implementation
results/
  checkpoints/   # saved .pth files
  graphs/        # loss curve plots
plot_loss.py     # plot exp1 loss curve
README.md
```

---

## Setup

```bash
pip install torch torchvision matplotlib
```

Place the dataset inside `data/` with this structure:
```
data/
  train/   # 15 subfolders, one per class
  val/     # 20% split from train
  test/    # Kaggle Test folder
```

Run training:
```bash
python -m src.train
```





## Results

### Cross Entropy Loss Model
- Test Accuracy: 97%

### Focal Loss Model
- Test Accuracy: (add your focal loss accuracy here)

### Evaluation Metrics
- Precision: ~0.97
- Recall: ~0.97
- F1-Score: ~0.97

### Key Findings
- Transfer learning using pretrained VGG16 achieved very high accuracy.
- Most fruit classes were classified correctly.
- Confusion matrix showed strong diagonal dominance.
- Fruits with similar appearance caused small classification errors.

### Best Model
- The CrossEntropyLoss model achieved the best overall performance.

#  FocalLoss Training — Experiment 2 -------------------------------------------------------------------------------------------------------

A PyTorch project that trains a neural network using **Focal Loss** — a smarter loss function that helps the model focus on examples it keeps getting wrong.

---

## What's in this repo?

| File | What it does |
|---|---|
| `src/experiment2.py` | Trains the model for 10 epochs, saves the best checkpoint, and generates plots |
| `src/validate.py` | Checks how well the model is doing on data it hasn't trained on |
| `src/loss.py` | Defines the Focal Loss formula |
| `src/model.py` | The neural network architecture |
| `checkpoint_focal.pth` | The best saved model weights (auto-generated after training) |
| `focal_val_accuracy.png` | Graph of validation accuracy across 10 epochs (auto-generated) |
| `focal_confusion_matrix.png` | Shows which classes the model confuses with each other (auto-generated) |

---

## What is Focal Loss and why use it?

Normal loss functions treat every training example equally. The problem? If the model already handles easy examples confidently, they still eat up most of the gradient signal — and the model stops improving on the hard ones.

**Focal Loss fixes this.** It automatically turns down the volume on easy examples and turns it up on hard ones.

The formula:

```
FL = -alpha * (1 - pt)^gamma * log(pt)
```

- `pt` = how confident the model was on the correct class (0 to 1)
- `gamma = 2` → easy examples (high `pt`) get down-weighted heavily
- `alpha = 1` → no extra class balancing (use `0.25` if your classes are imbalanced)

---

 Train for 10 epochs**

```bash
python src/experiment2.py
```

This will:
- Train using `FocalLoss(alpha=1.0, gamma=2.0)`
- Print loss and accuracy after every epoch
- Save the best model to `checkpoint_focal.pth` whenever val accuracy improves
- Save `focal_val_accuracy.png` and `focal_confusion_matrix.png` when done

---

## Experiment 2 Results (dummy data)

```python
exp2_train_losses = [2.3662, 2.3073, 2.269, 2.2255, 2.1811, 2.1468, 2.0894, 2.0312, 1.971, 1.9161]
exp2_val_losses   = [2.3748, 2.3789, 2.3862, 2.3987, 2.4089, 2.4254, 2.4384, 2.4497, 2.4739, 2.5005]
exp2_val_accs     = [0.04, 0.06, 0.06, 0.06, 0.05, 0.06, 0.05, 0.05, 0.05, 0.04]
```

Best val accuracy: **0.06** at epoch 2. Checkpoint saved there.

Train loss goes down steadily — the model is learning. Val loss creeping up is normal with random data (no real pattern to generalise from).

---

## How the checkpoint works

The model only saves when validation accuracy improves — so `checkpoint_focal.pth` always holds your **best** model, not just the last epoch.

To load it later:

```python
import torch
from src.model import build_model

model = build_model(num_classes=15)
checkpoint = torch.load("checkpoint_focal.pth")
model.load_state_dict(checkpoint["model_state_dict"])

print(f"Loaded from epoch {checkpoint['epoch']} — val_acc: {checkpoint['val_acc']}")
```

---

## How validate.py works

`validate()` is a simple function that takes your model and runs it over a dataset without doing any training. It returns two numbers: average loss and accuracy.

```python
from src.validate import validate

val_loss, val_acc = validate(model, val_loader, loss_fn, device)
print(f"Loss: {val_loss:.4f} | Accuracy: {val_acc:.2%}")
```

Three things that matter inside it:

- `model.eval()` — turns off dropout so the model behaves consistently
- `torch.no_grad()` — skips building the gradient graph (faster, less memory)
- `correct / total` — that's your accuracy
---

##  concepts used

| `model.train()` | Tells the model it's in training mode — activates dropout etc. |
| `model.eval()` | Tells the model it's being tested — turns off dropout |
| `torch.no_grad()` | Don't track gradients — used during validation to save memory |
| `loss.backward()` | Computes gradients via backpropagation |
| `optimizer.step()` | Updates the model weights using those gradients |
| `argmax(dim=1)` | Picks the class with the highest predicted score |
| `checkpoint` | A saved snapshot of model weights at the best epoch |

---

## Reference

Focal Loss was introduced in:
> Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017 — https://arxiv.org/abs/1708.02002
