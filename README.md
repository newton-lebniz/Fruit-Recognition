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