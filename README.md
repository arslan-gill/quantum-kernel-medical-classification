# Quantum Kernel Methods for Small-Sample Medical Classification

A comparative study of quantum vs. classical kernels on the Breast Cancer Wisconsin (Diagnostic) dataset.

## Overview

This project asks a direct, falsifiable question: **does a quantum fidelity kernel, built from an entangling feature map, capture structure in a real medical dataset that classical kernels (RBF, polynomial, linear) miss — and if not, why?**

Rather than assuming a quantum advantage, the project treats it as an empirical question and uses two diagnostic tools — **kernel-target alignment** and **circuit expressibility** — to explain the result rather than just report it.

**Headline result:** the quantum kernel underperformed every classical baseline by ~25–28 percentage points. Full analysis and discussion are in [`REPORT.md`](REPORT.md).

## Method Summary

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), 30 features, binary malignant/benign labels. A balanced 120-sample subset (60/60) keeps exact statevector simulation of the quantum kernel tractable.
- **Dimensionality reduction:** Standardized features, PCA to 4 components (82.8% variance retained), rescaled to [0, π] for angle encoding.
- **Quantum feature map:** ZZ-FeatureMap-style circuit (Hadamard + RZ data encoding + CNOT-RZ-CNOT entangling layer, 2 repetitions), implemented in PennyLane with exact statevector simulation (`default.qubit`). Linear and full entanglement topologies are both tested.
- **Quantum kernel:** Fidelity kernel k(x,y) = |⟨φ(x)|φ(y)⟩|², computed via full statevector overlap (no shot noise).
- **Classifier:** SVM with `kernel="precomputed"` fed the quantum Gram matrix directly.
- **Classical baselines:** SVM (RBF, polynomial degree 3, linear) and a 2-layer MLP (16, 16), evaluated identically via 5-fold stratified cross-validation with a fixed random seed.
- **Diagnostics:** Kernel-target alignment (Cristianini et al., 2001) and expressibility via KL divergence from the Haar-random fidelity distribution (Sim, Johnson & Aspuru-Guzik, 2019).

## Results

| Model | CV Accuracy | CV F1 | Kernel-Target Alignment |
|---|---|---|---|
| Neural Network (MLP, 2×16) | **0.958 ± 0.046** | 0.960 | n/a |
| Classical SVM (linear) | 0.950 ± 0.031 | 0.953 | n/a |
| Classical SVM (polynomial, deg 3) | 0.942 ± 0.057 | 0.944 | n/a |
| Classical SVM (RBF) | 0.933 ± 0.062 | 0.937 | n/a |
| Quantum Kernel SVM (linear entanglement) | 0.683 ± 0.068 | 0.672 | 0.0835 |
| Quantum Kernel SVM (full entanglement) | 0.675 ± 0.072 | 0.656 | 0.0815 |

Full numbers are in [`results.json`](results.json) / [`results_summary.csv`](results_summary.csv); figures are in `fig1`–`fig4`.

## Repository Structure

| File | Contents |
|---|---|
| `REPORT.md` | Full write-up: method, results, diagnostic analysis, and honest conclusion |
| `quantum_kernel.py` | Quantum feature map, fidelity kernel, alignment, and expressibility implementations |
| `run_experiment.py` | Data loading, PCA, and the cross-validated benchmark (quantum + classical models) |
| `make_plots.py` | Generates all four figures from the experiment results |
| `results.json`, `results_summary.csv` | Full numeric results for all six models |
| `fig1_kernel_matrices.png` | Quantum vs. classical kernel Gram matrices |
| `fig2_accuracy_comparison.png` | CV accuracy across all models |
| `fig3_expressibility.png` | Fidelity distributions vs. Haar-random reference |
| `fig4_alignment_vs_accuracy.png` | Kernel-target alignment as a predictor of accuracy |

## Requirements

```
numpy
pandas
scikit-learn
pennylane
matplotlib
```

Install with:

```bash
pip install numpy pandas scikit-learn pennylane matplotlib
```

## Running the Project

```bash
# Run the full benchmark (quantum + classical models, 5-fold CV)
python run_experiment.py

# Generate the figures from the results
python make_plots.py
```

`run_experiment.py` writes `results.json` and `results_summary.csv`. `make_plots.py` reads those outputs to produce `fig1`–`fig4`.

## Key Finding

Kernel-target alignment for both entanglement topologies was low (≈0.08), well below the ≈0.2–0.3 typical of a well-aligned kernel — meaning the quantum kernel's similarity structure barely tracked the true class labels, independent of classifier tuning. Notably, the *full*-entanglement circuit was **more expressible** (lower KL divergence: 0.086 vs. 0.128) yet performed marginally **worse**, consistent with literature on expressibility/task-usefulness tradeoffs and exponential concentration in quantum kernel methods (Thanasilp et al., 2022). See `REPORT.md` §4 for the full diagnostic discussion.

## Limitations

- Simulation only — no hardware noise modeling.
- Small qubit count (4), limited by classical simulation cost of the full Gram matrix.
- Fixed (non-trainable) feature map — a natural next step is a variational quantum kernel trained via kernel-target alignment.
- Single dataset — testing on data with known non-classical structure would isolate whether the tooling is correct versus whether the dataset/feature-map match is the limiting factor.

## References

- Havlicek, V. et al. (2019). *Supervised learning with quantum-enhanced feature spaces.* Nature.
- Schuld, M. & Killoran, N. (2019). *Quantum machine learning in feature Hilbert spaces.* PRL.
- Cristianini, N. et al. (2001). *On kernel-target alignment.* NeurIPS.
- Sim, S., Johnson, P.D. & Aspuru-Guzik, A. (2019). *Expressibility and entangling capability of parameterized quantum circuits.* Adv. Quantum Technol.
- Thanasilp, S. et al. (2022). *Exponential concentration in quantum kernel methods.*
- Hubregtsen, T. et al. (2022). *Training quantum embedding kernels on near-term quantum computers.*
