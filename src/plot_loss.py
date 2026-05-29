import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

exp1_train_losses = [0.0579,0.0556, 0.0461,0.0459,0.0420,0.0349,0.0281,0.0328,0.0325,0.0263]

if not exp1_train_losses:
    print("No losses yte paste your exp1_train_losses list at the top of this file")
    exit()

epochs = list(range(1, len(exp1_train_losses) + 1))
best_epoch = int(np.argmin(exp1_train_losses)) +1
best_loss = min(exp1_train_losses)

#plot
fig, ax = plt.subplots(figsize =(9,5))

ax.plot(
    epochs, exp1_train_losses,
    marker="o", linewidth=2, color="#2563EB",
    markerfacecolor="white", markeredgewidth=2, markersize=7,
    label="Train Loss" 
)

#shaded area under curve
ax.fill_between(epochs, exp1_train_losses, alpha=0.08, color="#2563EB")
 
# Annotate best (lowest) loss point
ax.annotate(
    f"Best: {best_loss:.4f}\n(epoch {best_epoch})",
    xy=(best_epoch, best_loss),
    xytext=(best_epoch + 0.6, best_loss + 0.004),
    arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.2),
    fontsize=9, color="#64748B"
)

# lables and styling
ax.set_title(
    "Experiment 1 — Training Loss over Epochs\n"
    "VGG16 (frozen features) · CrossEntropyLoss · Adam lr=1e-4",
    fontsize=12, fontweight="bold", pad=12
)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Average Training Loss", fontsize=11)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xlim(0.5, len(epochs) + 0.5)
ax.set_ylim(0, max(exp1_train_losses) * 1.25)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
 
plt.tight_layout()

#save
os.makedirs("results/graphs", exist_ok=True)
out = "results/graphs/exp1_loss_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")