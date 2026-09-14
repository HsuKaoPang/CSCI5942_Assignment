import json
import matplotlib.pyplot as plt

models = ["pico", "nano", "micro", "mini", "base", "base-tinystories"]
corpora = ["shakespeare", "tinystories", "wikipedia", "python-code"]

data = []
for m in models:
    f = open(f"results/{m}.json")
    r = json.load(f)
    row = [r["nats_per_char"][c] for c in corpora]
    data.append(row)
    f.close()

plt.figure(figsize=(8, 6))
plt.imshow(data, cmap="viridis_r")
plt.xticks(range(len(corpora)), corpora)
plt.yticks(range(len(models)), models)

for i in range(len(models)):
    for j in range(len(corpora)):
        plt.text(j, i, str(data[i][j]), ha="center", va="center", color="black")

plt.colorbar()
plt.title("nats/char, models x corpora (with twin)")
plt.tight_layout()
plt.savefig("results/heatmap_twin.png")
print("saved results/heatmap_twin.png")
