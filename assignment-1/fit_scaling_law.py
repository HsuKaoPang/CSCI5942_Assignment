"""Fit L(N) = c + a * N^-alpha to the scaling-ladder results and plot it.

Reads out/<name>/summary.json for each run in RUNS, fits the saturating
power law with scipy, and saves a log-log plot with the four data
points plus the fitted curve.

    python fit_scaling_law.py
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["nano", "micro", "mini", "base"]


def power_law(N, c, a, alpha):
    return c + a * N ** (-alpha)


def main():
    Ns, losses = [], []
    for name in RUNS:
        with open(os.path.join(HERE, "out", name, "summary.json")) as f:
            s = json.load(f)
        Ns.append(s["n_params"])
        losses.append(s["best_val_loss"])
        print(f"{name:>6}: N={s['n_params']:>10,}  best_val_loss={s['best_val_loss']:.4f}")

    Ns = np.array(Ns, dtype=np.float64)
    losses = np.array(losses, dtype=np.float64)

    # Initial guesses: c a bit below the smallest observed loss, alpha ~ 0.3
    # (typical scaling-law exponents are well under 1), bounds keep the fit
    # in the physically sensible region (c >= 0, a > 0, alpha > 0).
    p0 = [losses.min() * 0.9, 1.0, 0.3]
    bounds = ([0, 0, 0], [losses.min(), np.inf, 5])
    (c, a, alpha), pcov = curve_fit(power_law, Ns, losses, p0=p0, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    print()
    print(f"fitted: c={c:.4f} (+/-{perr[0]:.4f})  a={a:.4f} (+/-{perr[1]:.4f})  "
          f"alpha={alpha:.4f} (+/-{perr[2]:.4f})")

    N_smooth = np.logspace(np.log10(Ns.min()) - 0.3, np.log10(Ns.max()) + 0.3, 200)
    L_smooth = power_law(N_smooth, c, a, alpha)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(N_smooth, L_smooth, "-", color="C0", label=f"fit: c={c:.3f}, a={a:.3f}, α={alpha:.3f}")
    ax.scatter(Ns, losses, color="C1", zorder=5, label="observed (N, best val loss)")
    for name, N, L in zip(RUNS, Ns, losses):
        ax.annotate(name, (N, L), textcoords="offset points", xytext=(6, 6))
    ax.axhline(c, color="gray", linestyle=":", linewidth=1, label=f"floor c={c:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (non-embedding parameters)")
    ax.set_ylabel("best val loss")
    ax.set_title("Scaling law fit: L(N) = c + a * N^-alpha")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(HERE, "out", "scaling_law.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved plot to {out_path}")


if __name__ == "__main__":
    main()
