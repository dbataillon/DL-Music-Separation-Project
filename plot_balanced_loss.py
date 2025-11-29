import json
import os
import matplotlib.pyplot as plt

MODEL_DIR = "models/unet_balanced"
LOSS_LOG = os.path.join(MODEL_DIR, "training_loss.json")

if not os.path.exists(LOSS_LOG):
    raise FileNotFoundError(f"Could not find loss log at {LOSS_LOG}")

with open(LOSS_LOG, "r") as f:
    loss_history = json.load(f)

epochs = []
losses = []

for entry in loss_history.values():
    epochs.append(entry["epoch"])
    losses.append(entry["loss"])

epochs, losses = zip(*sorted(zip(epochs, losses)))

plt.figure(figsize=(10, 4))
plt.plot(epochs, losses, marker="o", markersize=2, linewidth=1)
plt.xlabel("Epoch")
plt.ylabel("Average L1 Loss")
plt.title("Balanced U-Net Training Loss per Epoch")
plt.grid(True, alpha=0.3)

out_path = os.path.join(MODEL_DIR, "balanced_unet_training_loss.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
