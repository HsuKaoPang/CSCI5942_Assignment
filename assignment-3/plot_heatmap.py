import json
import sys
import glob
import os
import matplotlib.pyplot as plt

# mixes to plot, in order. Pass them as arguments, or let it pick up
# whatever eval/results holds with baseline and english_pages first.
mixes = sys.argv[1:]
if not mixes:
    found = sorted(os.path.basename(p)[:-5]
                   for p in glob.glob("eval/results/*.json"))
    first = [m for m in ("baseline", "english_pages") if m in found]
    mixes = first + [m for m in found if m not in first]

corpora = ["wikipedia", "tinystories", "shakespeare", "python-code"]

data = []
for m in mixes:
    f = open(f"eval/results/{m}.json")
    r = json.load(f)
    row = [r["nats_per_char"][c] for c in corpora]
    data.append(row)
    f.close()

plt.figure(figsize=(8, 6), layout="constrained")
# aspect="auto" lets the 4 columns stretch to fill the axes; the default
# "equal" makes square cells, which leaves a tall narrow plot stranded
# against the right-hand side of a wide figure.
plt.imshow(data, cmap="viridis_r", aspect="auto")
plt.xticks(range(len(corpora)), corpora, rotation=20, ha="right")
plt.yticks(range(len(mixes)), mixes)

for i in range(len(mixes)):
    for j in range(len(corpora)):
        plt.text(j, i, str(data[i][j]), ha="center", va="center", color="black")

plt.colorbar(shrink=0.85)
plt.title("nats/char, mixes x corpora")
plt.savefig("eval/results/heatmap.png")
print("saved eval/results/heatmap.png")
