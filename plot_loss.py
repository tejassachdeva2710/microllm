"""plot_loss.py - reads checkpoints_v2/loss_log.csv, writes loss_curve.png. Needs: pip install matplotlib"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = r"D:\micro-llm\checkpoints_v2\loss_log.csv"
steps, tr, va = [], [], []
with open(log) as f:
    for row in csv.DictReader(f):
        steps.append(int(row["step"])); tr.append(float(row["train_loss"])); va.append(float(row["val_loss"]))

plt.figure(figsize=(9, 5))
plt.plot(steps, tr, label="train loss")
plt.plot(steps, va, label="validation loss")
plt.xlabel("step (24,576 tokens/step)"); plt.ylabel("cross-entropy loss")
plt.title("22.5M-param transformer trained from scratch on an RTX 3050 Ti laptop")
plt.grid(alpha=0.3); plt.legend()
out = r"D:\micro-llm\loss_curve.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}  (final val loss {va[-1]:.4f} at step {steps[-1]})")
