import csv
import matplotlib.pyplot as plt

steps = []
train_loss = []
val_loss = []

f = open("../assignment-1/out/base-tinystories/log.csv")
for row in csv.DictReader(f):
    steps.append(int(row["step"]))
    train_loss.append(float(row["train_loss"]))
    val_loss.append(float(row["val_loss"]))
f.close()

plt.figure(figsize=(8, 6))
plt.plot(steps, train_loss, label="train")
plt.plot(steps, val_loss, label="val")
plt.xlabel("step")
plt.ylabel("loss (nats/char)")
plt.title("twin (base config on TinyStories) train/val curves")
plt.legend()
plt.tight_layout()
plt.savefig("results/twin_curves.png")
print("saved results/twin_curves.png")
