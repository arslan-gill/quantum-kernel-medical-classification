# Quantum Kernel Methods for Small-Sample Medical Classification

**Benchmarking a Fidelity Quantum Kernel Against Classical Kernels, with Kernel-Target Alignment and Expressibility Diagnostics**

Prepared by Arslan Gill
September 2026

## Abstract

Quantum kernel methods have been proposed as a route to a practical quantum advantage in
classification, on the theoretical premise that an entangling feature map can project data into a
Hilbert space where classically-hard-to-separate structure becomes linearly separable [1, 2]. This
project implements a fidelity-based quantum kernel in PennyLane and benchmarks it directly against
classical kernel methods (RBF, polynomial, linear SVM) and a classical neural network, on a
120-sample, class-balanced subset of the Breast Cancer Wisconsin (Diagnostic) dataset, reduced via
PCA to 4 dimensions for 4-qubit angle encoding. Two entangling topologies were tested — linear
(nearest-neighbor) and full (all-to-all) CNOT connectivity — using a ZZ-FeatureMap-style circuit
with exact statevector fidelity computation (no shot noise). Under 5-fold stratified cross-validation,
both quantum kernels underperformed every classical baseline by a wide margin: 68.3% (linear
entanglement) and 67.5% (full entanglement) accuracy, versus 93.3%–95.8% for the classical SVMs
and MLP. Rather than treating this as a dead end, the project uses two diagnostic tools — kernel-target
alignment [3] and circuit expressibility [4] — to explain the gap. Alignment scores were low
(0.081–0.083) for both circuits, indicating the quantum kernel's notion of similarity correlates
weakly with the true class labels regardless of downstream classifier tuning. Expressibility analysis
shows the more entangled (full-connectivity) circuit is measurably closer to Haar-random coverage of
Hilbert space (KL divergence 0.086 vs. 0.128) yet performs marginally worse — evidence that
expressibility and task-usefulness are not the same property, consistent with exponential-concentration
effects reported in recent QML literature [5]. The negative result is reported directly rather than
tuned away; Section 6 discusses what would be required for a fairer test of quantum advantage on this
class of problem.

## 1. Motivation and Background

Variational and kernel-based quantum machine learning (QML) methods are among the most concrete
near-term applications proposed for NISQ-era devices. Havlicek et al. [1] showed that a quantum
feature map can, in principle, define a kernel that is classically hard to estimate, and demonstrated
supervised learning with such a feature space on small synthetic and real datasets. Schuld and
Killoran [2] formalized the broader view of quantum models as kernel methods operating in a
quantum-enhanced feature Hilbert space, unifying variational and kernel-based QML under one
framework.

A recurring open question in this literature is *when* a quantum kernel actually helps: the existence
proof that quantum feature maps can define classically-intractable kernels does not imply that an
arbitrary quantum feature map improves classification on an arbitrary classical dataset. This project
treats that as an empirical question rather than an assumption, and — following Cristianini et al. [3]
and Sim, Johnson, and Aspuru-Guzik [4] — uses kernel-target alignment and circuit expressibility as
diagnostic tools to explain *why* a given quantum kernel does or does not perform well, rather than
reporting an accuracy number in isolation.

**Project goal:** implement a fidelity-based quantum kernel, benchmark it rigorously against classical
kernel methods on a real (not synthetic) medical dataset under identical cross-validation, and use
alignment/expressibility diagnostics to characterize the result — positive or negative — with the same
rigor either way.

## 2. Methods

### 2.1 Dataset and preprocessing

The Breast Cancer Wisconsin (Diagnostic) dataset (30 features, binary malignant/benign labels) was
used. A class-balanced 120-sample subset (60/60) was drawn to keep exact statevector simulation of
the quantum kernel tractable, since each Gram matrix entry requires simulating two 4-qubit circuits
and the full kernel matrix is O(n²) in sample count. Features were standardized, reduced via PCA to
4 components (82.8% of variance retained), and rescaled to [0, π] for angle encoding — one feature
per qubit.

### 2.2 Quantum feature map and kernel

A ZZ-FeatureMap-style circuit [1] was implemented in PennyLane: each of 2 repetitions applies a
Hadamard layer, RZ data-encoding rotations, and a CNOT–RZ–CNOT entangling layer across qubit
pairs. Two entangling topologies were compared: **linear** (nearest-neighbor pairs only) and **full**
(all-to-all pairs). The kernel is the fidelity between encoded states,

k(x, y) = |⟨φ(x)|φ(y)⟩|²,

computed via exact statevector simulation (`default.qubit`) rather than shot-based sampling, isolating
the feature map's intrinsic properties from sampling noise.

### 2.3 Classifier

The quantum Gram matrix was passed directly to a support vector machine via
`SVC(kernel="precomputed")` — the standard way to combine a quantum kernel with a classical
optimizer [1, 2].

### 2.4 Classical baselines

SVMs with RBF, polynomial (degree 3), and linear kernels, and a 2-hidden-layer (16, 16) multilayer
perceptron, were trained on the identical PCA-reduced features under the identical cross-validation
protocol.

### 2.5 Diagnostics

- **Kernel-target alignment** [3]: A(K, yyᵀ) = ⟨K, yyᵀ⟩_F / (‖K‖_F ‖yyᵀ‖_F), measuring how well a
  kernel's similarity structure agrees with the true labels, independent of the downstream classifier.
- **Circuit expressibility** [4]: KL divergence between the empirical fidelity distribution of the
  feature map (sampled over random input pairs) and the theoretical Haar-random fidelity
  distribution. Lower KL divergence indicates a more expressive circuit, i.e. one whose outputs more
  uniformly cover the accessible Hilbert space.

### 2.6 Evaluation protocol

All six models (2 quantum kernel variants, 3 classical SVM kernels, 1 neural network) were evaluated
under identical 5-fold stratified cross-validation, same random seed, same train/test splits, reporting
mean accuracy and F1 with standard deviation across folds.

## 3. Results

### 3.1 Headline numbers

| Model | CV Accuracy | CV F1 | Kernel-Target Alignment | Expressibility (KL) |
|---|---|---|---|---|
| Neural Network (MLP, 2×16) | **0.958 ± 0.046** | 0.960 | n/a | n/a |
| Classical SVM (linear) | 0.950 ± 0.031 | 0.953 | n/a | n/a |
| Classical SVM (polynomial, deg 3) | 0.942 ± 0.057 | 0.944 | n/a | n/a |
| Classical SVM (RBF) | 0.933 ± 0.062 | 0.937 | n/a | n/a |
| Quantum Kernel SVM (linear entanglement) | 0.683 ± 0.068 | 0.672 | 0.0835 | 0.1276 |
| Quantum Kernel SVM (full entanglement) | 0.675 ± 0.072 | 0.656 | 0.0815 | 0.0858 |

*(Full numeric results in `results.json` / `results_summary.csv`.)*

The quantum kernel underperformed every classical method by 25–28 percentage points in
cross-validated accuracy, with both entanglement topologies scoring similarly (68.3% vs. 67.5%) —
the choice of entangling connectivity did not meaningfully change the outcome.

### 3.2 Visual comparison

*Figure 1. Kernel Gram matrices — quantum (linear), quantum (full), and classical RBF — samples
sorted by class, dashed line marks the class boundary. The classical RBF kernel shows visible block
structure aligned with class boundaries; the quantum kernels show comparatively little.*

*Figure 2. Cross-validated accuracy across all six models, error bars from 5-fold standard deviation.
Chance level (0.5) marked for reference.*

*Figure 3. Empirical pairwise-fidelity distribution of both quantum feature maps against the
theoretical Haar-random distribution — the closer a histogram sits to the dashed reference curve, the
more expressible the circuit.*

*Figure 4. Kernel-target alignment plotted against cross-validated accuracy for both quantum kernel
variants — both metrics move together in this small comparison, consistent with alignment being a
useful low-cost proxy for a kernel's eventual classification performance.*

### 3.3 Why the quantum kernel underperformed

Kernel-target alignment was low (≈0.08) for both entanglement topologies. For reference, a
well-aligned kernel on a task of this kind typically scores well above 0.2–0.3. This is diagnostic
*before* any classifier is trained: it indicates the quantum kernel's notion of similarity barely
correlates with the true class labels, so no amount of SVM hyperparameter tuning could be expected to
recover strong performance from this Gram matrix.

Expressibility analysis (Figure 3) clarifies why. The full-entanglement circuit is measurably *more*
expressible than the linear circuit (KL divergence 0.086 vs. 0.128, closer to the Haar-random
reference) — yet it performs marginally *worse* (67.5% vs. 68.3% accuracy). This is a known and
important effect in the QML literature: expressibility and task-usefulness are not the same property.
A highly expressible circuit spreads encoded data too uniformly across Hilbert space, which can
destroy label-relevant structure rather than reveal it — related to the exponential-concentration
phenomena described by Thanasilp et al. [5]. More entanglement does not automatically produce a
better kernel for a fixed classical dataset.

The kernel matrix visualization (Figure 1) confirms this directly: values in the quantum kernels are
high and comparatively uniform across both classes, consistent with the low alignment score, whereas
the classical RBF kernel shows clear block structure matching the class boundary.

**Most likely root cause:** with only 4 qubits and a fixed, generic (not task-tuned) feature map, the
data encoding is too generic and the entangling layer too aggressive relative to the amount of
task-relevant structure present in 4 PCA components of this dataset.

## 4. Discussion

This project does not demonstrate a quantum advantage on this dataset, and reports that directly
rather than adjusting the framing after the fact. What it does demonstrate is a correctly implemented
quantum kernel method, a fair and identically-controlled comparison against multiple classical
baselines, and — more importantly — a diagnostic explanation for the result rather than a bare accuracy
table. The finding that a *more* expressible circuit did not yield *better* alignment or accuracy is
itself a useful, literature-consistent negative result: it is evidence against the naive assumption that
"more entanglement is better" for a fixed quantum kernel on classical tabular data, and points toward
task-adapted or trainable feature maps as the more promising direction, discussed in Section 6.

## 5. Reproducibility

All code, results, and figures are included alongside this report. The pipeline is fully deterministic
given the fixed random seed used for data subsampling and cross-validation.

- `quantum_kernel.py` — quantum feature map, fidelity kernel, kernel-target alignment, expressibility
- `run_experiment.py` — data loading, PCA reduction, cross-validated benchmark (quantum + classical)
- `make_plots.py` — generates all four figures
- `results.json`, `results_summary.csv` — full numeric results

## 6. Limitations and Threats to Validity

- **Simulation only**: no hardware noise modeling. Exact statevector fidelity was used specifically to
  isolate the feature map's intrinsic properties from sampling/hardware noise; a real-device run would
  be expected to perform worse still.
- **Small qubit count (4)**: limited by the classical simulation cost of the full O(n²) Gram matrix.
  Retaining more of the original 30 features (more qubits, or approximate kernel estimation via random
  Fourier features or sparse kernel methods) was not evaluated.
- **Fixed feature map**: the ZZ-FeatureMap used has no trainable parameters. The natural next step is a
  *trainable* quantum kernel, with feature-map parameters optimized directly against kernel-target
  alignment as the training objective [6], rather than a fixed, generic encoding.
- **Single dataset**: testing on data with a known non-classical structure (e.g. parity or graph-based
  synthetic tasks where a quantum kernel advantage is provable) was not performed, and would help
  isolate whether the tooling is correct and only the dataset/feature-map match is the limiting factor
  here.

## 7. Conclusion

A fidelity-based quantum kernel, implemented in PennyLane and evaluated under identical
cross-validation to four classical baselines on a real medical dataset, underperformed every classical
method by a wide margin (67.5–68.3% vs. 93.3–95.8% accuracy). Kernel-target alignment and circuit
expressibility diagnostics indicate this is not simply an undertrained classifier problem: the quantum
kernel's similarity structure correlates weakly with the true labels regardless of entangling topology,
and higher circuit expressibility did not translate into better task performance. The clearest next
steps are evaluating a trainable quantum kernel optimized directly against alignment, and testing on a
dataset with structure more suited to entangling feature maps.

## References

[1] V. Havlicek, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta,
"Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, pp. 209–212, 2019.

[2] M. Schuld and N. Killoran, "Quantum machine learning in feature Hilbert spaces," *Physical Review
Letters*, vol. 122, 040504, 2019.

[3] N. Cristianini, J. Shawe-Taylor, A. Elisseeff, and J. Kandola, "On kernel-target alignment," in
*Advances in Neural Information Processing Systems (NeurIPS)*, vol. 14, 2001.

[4] S. Sim, P. D. Johnson, and A. Aspuru-Guzik, "Expressibility and entangling capability of
parameterized quantum circuits for hybrid quantum-classical algorithms," *Advanced Quantum
Technologies*, vol. 2, 1900070, 2019.

[5] S. Thanasilp, S. Wang, N. A. Nghiem, P. J. Coles, and M. Cerezo, "Exponential concentration in
quantum kernel methods," *arXiv:2208.11060*, 2022.

[6] T. Hubregtsen, D. Wierichs, E. Gil-Fuster, P.-J. H. S. Derks, P. K. Faehrmann, and J. J. Meyer,
"Training quantum embedding kernels on near-term quantum computers," *Physical Review A*, vol. 106,
042431, 2022.

[7] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning
Research*, vol. 12, pp. 2825–2830, 2011.

[8] Xanadu, "PennyLane: An open-source framework for quantum machine learning and quantum
computing," https://pennylane.ai.
