import matplotlib.pyplot as plt
import os

exp1_train_losses = [0.0579,0.0556, 0.0461,0.0459,0.0420,0.0349,0.0281,0.0328,0.0325,0.0263]

if not exp1_train_losses:
    print("No losses yte paste your exp1_train_losses list at the top of this file")
    exit()

epochs = list(range(1, len(exp1_train_losses) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, exp1_train_losses, marker='o', linewidth=2,
         color='steelblue', label='Train Loss (CrossEntropy)')
 
plt.title("Experiment 1 — Loss Curve (VGG16, CrossEntropyLoss)", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.xticks(epochs)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
 
os.makedirs("results/graphs", exist_ok=True)
plt.savefig("results/graphs/exp1_loss_curve.png", dpi=150)
print("Saved → results/graphs/exp1_loss_curve.png")
plt.show()