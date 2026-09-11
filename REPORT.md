# Quantum Kernel Methods for Small-Sample Medical Classification

**A comparative study of quantum vs. classical kernels on the Breast Cancer Wisconsin (Diagnostic) dataset**

---

## 1. Research Question

Small-sample, moderate-dimensional medical classification tasks are a setting where quantum
kernel methods are theoretically argued to offer an advantage: entangling feature maps can, in
principle, project data into a Hilbert space where classically-hard-to-separate structure becomes
linearly separable (Havlicek et al., 2019; Schuld & Killoran, 2019). This project asks directly:

> **Does a quantum fidelity kernel, built from an entangling feature map, capture structure in a
> real medical dataset that classical kernels (RBF, polynomial, linear) miss — and if not, why?**

Rather than assuming a quantum advantage, this project treats it as an empirical question and uses
two diagnostic tools — **kernel-target alignment** and **circuit expressibility** — to explain the
result rather than just report it.

## 2. Method

**Dataset.** Breast Cancer Wisconsin (Diagnostic), 30 features, binary malignant/benign labels.
A balanced 120-sample subset was used (60/60) to keep exact statevector simulation of the quantum
kernel tractable — each kernel entry requires simulating two 4-qubit circuits, and the full
Gram matrix is O(n²).

**Dimensionality reduction.** Features were standardized and reduced via PCA to 4 components
(82.8% variance retained), then rescaled to [0, π] for angle encoding — one feature per qubit.

**Quantum feature map.** A ZZ-FeatureMap-style circuit (Hadamard layer + RZ data encoding +
CNOT-RZ-CNOT entangling layer, 2 repetitions), implemented in PennyLane with exact statevector
simulation (`default.qubit`). Two entanglement topologies were tested: **linear** (nearest-neighbor)
and **full** (all-to-all).

**Quantum kernel.** Fidelity kernel k(x,y) = |⟨φ(x)|φ(y)⟩|², computed via full statevector overlap
(no shot noise — this isolates the feature map's intrinsic properties from sampling error).

**Classifier.** SVM with `kernel="precomputed"`, fed the quantum Gram matrix directly — this is
the standard way to plug a quantum kernel into a classical SVM (quantum kernel estimation +
classical optimization, per Havlicek et al.).

**Classical baselines.** SVM with RBF, polynomial (degree 3), and linear kernels; a 2-layer MLP
(16, 16 units). All models evaluated identically: same PCA-reduced features, same 5-fold stratified
cross-validation, same random seed.

**Diagnostics.**
- *Kernel-target alignment* (Cristianini et al., 2001) — measures how well a kernel's similarity
  structure agrees with the true labels, independent of the downstream classifier.
- *Expressibility* (Sim, Johnson & Aspuru-Guzik, 2019) — KL divergence between the empirical
  fidelity distribution of the feature map and the theoretical Haar-random distribution. Lower =
  more expressive circuit, covering more of Hilbert space.

## 3. Results

| Model | CV Accuracy | CV F1 | Kernel-Target Alignment |
|---|---|---|---|
| Neural Network (MLP, 2×16) | **0.958 ± 0.046** | 0.960 | n/a |
| Classical SVM (linear) | 0.950 ± 0.031 | 0.953 | n/a |
| Classical SVM (polynomial, deg 3) | 0.942 ± 0.057 | 0.944 | n/a |
| Classical SVM (RBF) | 0.933 ± 0.062 | 0.937 | n/a |
| Quantum Kernel SVM (linear entanglement) | 0.683 ± 0.068 | 0.672 | 0.0835 |
| Quantum Kernel SVM (full entanglement) | 0.675 ± 0.072 | 0.656 | 0.0815 |

*(Full numbers in `results.json` / `results_summary.csv`; figures in `fig1`–`fig4`.)*

**Headline result: the quantum kernel underperformed every classical method by ~25–28 percentage
points**, only modestly above chance-adjusted for a balanced binary task (50%).

## 4. Why — Diagnostic Analysis

This is the part that matters more than the accuracy table.

**Kernel-target alignment was low (≈0.08) for both entanglement topologies.** For comparison,
a well-aligned kernel on a task like this typically scores well above 0.2–0.3. This tells us
*before even training a classifier* that the quantum kernel's notion of "similarity" barely
correlates with the actual class labels — the SVM was working with a Gram matrix that doesn't
separate the classes well, regardless of tuning.

**Expressibility analysis (Figure 3) shows why.** The empirical fidelity distribution of both
feature maps sits well below the Haar-random reference curve, but more importantly: the *full*
entanglement circuit is **more** expressible (lower KL divergence, 0.086 vs 0.128) yet performs
marginally **worse**. This is a known and important effect in QML — expressibility and
task-usefulness are not the same thing. A highly expressible circuit spreads data too uniformly
across Hilbert space, which can *destroy* the label-relevant structure rather than reveal it
(this relates to the "exponential concentration" phenomena discussed in recent QML literature,
e.g. Thanasilp et al., 2022). More entanglement does not automatically mean a better kernel.

**Kernel matrix visualization (Figure 1)** confirms this directly: the classical RBF kernel shows
visible block structure aligned with the class boundary (dashed line); the quantum kernels show
comparatively little block structure — values are high and relatively uniform across both classes,
consistent with the low alignment score.

**Root cause, most likely:** with only 4 qubits and a generic (not task-tuned) feature map, the
encoding is too generic and the entangling layer too aggressive relative to the amount of
task-relevant structure in 4 PCA components of this dataset. This is consistent with a growing
body of QML literature suggesting that **naive quantum kernels do not automatically outperform
well-tuned classical kernels on classical data** — the advantage, where it exists, requires
either problem structure specifically suited to quantum feature maps or trainable
(variational) feature maps rather than fixed ones.

## 5. Honest Conclusion

This project does **not** demonstrate a quantum advantage — and reports that clearly rather than
overselling the result. What it does demonstrate:

1. A working, correct implementation of a quantum kernel method with exact fidelity computation.
2. A rigorous, apples-to-apples benchmark against multiple classical baselines.
3. Diagnostic tooling (alignment, expressibility) that explains *why* the result came out this
   way, rather than stopping at an accuracy number — this is the difference between a tutorial
   exercise and an analysis.
4. A concrete, literature-grounded hypothesis for the negative result (expressibility/alignment
   mismatch), and a clear next step: task-adapted or trainable quantum feature maps.

## 6. Limitations & Next Steps

- **Simulation only** — no hardware noise modeling; real NISQ devices would perform worse still.
- **Small qubit count (4)** — limited by classical simulation cost of the full Gram matrix; larger
  qubit counts or approximate kernel estimation (random Fourier features, sparse kernels) would
  let more variance be retained from the original 30 features.
- **Fixed feature map** — the natural next step is a *trainable* quantum feature map (e.g.
  variational parameters optimized via kernel-target alignment as the training objective, per
  Hubregtsen et al., 2022), rather than a fixed ZZ-FeatureMap.
- **Single dataset** — testing on data with known non-classical structure (e.g. specific parity
  or graph-based synthetic tasks where quantum kernels are proven to have advantage) would isolate
  whether the tooling is correct and only the dataset/feature-map match is the issue.

## 7. Files in this project

| File | Contents |
|---|---|
| `quantum_kernel.py` | Quantum feature map, fidelity kernel, alignment, expressibility |
| `run_experiment.py` | Data loading, PCA, cross-validated benchmark (quantum + classical) |
| `make_plots.py` | Generates all four figures |
| `results.json`, `results_summary.csv` | Full numeric results |
| `fig1_kernel_matrices.png` | Quantum vs classical kernel Gram matrices |
| `fig2_accuracy_comparison.png` | CV accuracy, all models |
| `fig3_expressibility.png` | Fidelity distributions vs Haar-random |
| `fig4_alignment_vs_accuracy.png` | Alignment as accuracy predictor |

---

### References
Havlicek, V. et al. (2019). *Supervised learning with quantum-enhanced feature spaces.* Nature.
Schuld, M. & Killoran, N. (2019). *Quantum machine learning in feature Hilbert spaces.* PRL.
Cristianini, N. et al. (2001). *On kernel-target alignment.* NeurIPS.
Sim, S., Johnson, P.D. & Aspuru-Guzik, A. (2019). *Expressibility and entangling capability of
parameterized quantum circuits.* Adv. Quantum Technol.
Thanasilp, S. et al. (2022). *Exponential concentration in quantum kernel methods.*
Hubregtsen, T. et al. (2022). *Training quantum embedding kernels on near-term quantum computers.*
