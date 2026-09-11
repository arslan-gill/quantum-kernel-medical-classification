import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

from run_experiment import load_and_prepare_data
from quantum_kernel import QuantumKernel, circuit_expressibility

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

with open("results.json") as f:
    results = json.load(f)

X, y, _, _ = load_and_prepare_data()
order = np.argsort(y)  # sort samples by class for clearer kernel heatmaps
X_sorted, y_sorted = X[order], y[order]

K_lin = np.load("kernel_matrix_linear.npy")[np.ix_(order, order)]
K_full = np.load("kernel_matrix_full.npy")[np.ix_(order, order)]
K_rbf = rbf_kernel(X_sorted, X_sorted)

# ---------- Figure 1: Kernel matrix heatmaps ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, K, title in zip(
    axes, [K_lin, K_full, K_rbf],
    ["Quantum Kernel\n(linear entanglement)", "Quantum Kernel\n(full entanglement)", "Classical RBF Kernel"]
):
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1)
    ax.axvline(59.5, color="white", lw=0.8, ls="--")
    ax.axhline(59.5, color="white", lw=0.8, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="kernel value")
fig.suptitle("Kernel Matrices (samples sorted by class; dashed line = class boundary)", y=1.02)
plt.savefig("fig1_kernel_matrices.png", bbox_inches="tight")
plt.close()

# ---------- Figure 2: CV accuracy comparison ----------
labels = [results[k]["label"] for k in results]
accs = [results[k]["cv_accuracy_mean"] for k in results]
stds = [results[k]["cv_accuracy_std"] for k in results]
colors = ["#7b2cbf" if "Quantum" in l else "#2077b4" for l in labels]

order_idx = np.argsort(accs)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([labels[i] for i in order_idx], [accs[i] for i in order_idx],
        xerr=[stds[i] for i in order_idx], color=[colors[i] for i in order_idx],
        capsize=4)
ax.set_xlabel("5-fold CV Accuracy")
ax.set_xlim(0, 1.05)
ax.set_title("Model Comparison: Quantum Kernel vs Classical Methods")
ax.axvline(0.5, color="gray", ls=":", lw=1, label="chance level")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig2_accuracy_comparison.png", bbox_inches="tight")
plt.close()

# ---------- Figure 3: Expressibility (fidelity distributions vs Haar) ----------
qk_lin = QuantumKernel(n_qubits=4, reps=2, entanglement="linear")
qk_full = QuantumKernel(n_qubits=4, reps=2, entanglement="full")
_, fid_lin = circuit_expressibility(qk_lin.feature_map, 4)
_, fid_full = circuit_expressibility(qk_full.feature_map, 4)

N = 2 ** 4
xs = np.linspace(0.001, 0.999, 300)
haar_pdf = (N - 1) * (1 - xs) ** (N - 2)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(fid_lin, bins=30, density=True, alpha=0.5, label="Linear entanglement", color="#7b2cbf")
ax.hist(fid_full, bins=30, density=True, alpha=0.5, label="Full entanglement", color="#e07a00")
ax.plot(xs, haar_pdf, "k--", lw=1.5, label="Haar-random (ideal)")
ax.set_xlabel("Pairwise fidelity |<phi(x)|phi(y)>|^2")
ax.set_ylabel("Density")
ax.set_title("Feature Map Expressibility\n(closer to dashed line = more expressive)")
ax.legend()
plt.tight_layout()
plt.savefig("fig3_expressibility.png", bbox_inches="tight")
plt.close()

# ---------- Figure 4: Alignment vs Accuracy ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
align_vals, acc_vals, lbls = [], [], []
for k, v in results.items():
    if v["kernel_target_alignment"] is not None:
        align_vals.append(v["kernel_target_alignment"])
        acc_vals.append(v["cv_accuracy_mean"])
        lbls.append(v["label"].replace("Quantum Kernel SVM ", ""))
ax.scatter(align_vals, acc_vals, color="#7b2cbf", s=80, zorder=3)
for x_, y_, l in zip(align_vals, acc_vals, lbls):
    ax.annotate(l, (x_, y_), textcoords="offset points", xytext=(8, -3), fontsize=9)
ax.set_xlabel("Kernel-Target Alignment")
ax.set_ylabel("CV Accuracy")
ax.set_title("Quantum Kernel: Alignment vs. Accuracy")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig4_alignment_vs_accuracy.png", bbox_inches="tight")
plt.close()

print("Saved: fig1_kernel_matrices.png, fig2_accuracy_comparison.png, "
      "fig3_expressibility.png, fig4_alignment_vs_accuracy.png")
