"""
Main experiment: Quantum Kernel vs Classical Methods
======================================================
Dataset: Breast Cancer Wisconsin (Diagnostic) -- a small, real, binary
medical classification dataset (569 samples, 30 features), reduced via
PCA to n_qubits dimensions so each feature maps cleanly onto one qubit.

Models compared:
  1. Quantum kernel SVM (fidelity kernel, linear entanglement)
  2. Quantum kernel SVM (fidelity kernel, full entanglement)
  3. Classical SVM -- RBF kernel
  4. Classical SVM -- polynomial kernel
  5. Classical SVM -- linear kernel
  6. Classical Neural Network (MLP)

Metrics: 5-fold cross-validated accuracy + F1, kernel-target alignment,
circuit expressibility, wall-clock training time.
"""

import time
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score

from quantum_kernel import QuantumKernel, kernel_target_alignment, circuit_expressibility

RANDOM_STATE = 42
N_QUBITS = 4          # dataset reduced to this many dimensions via PCA
N_FOLDS = 5
SUBSAMPLE_N = 120      # keep quantum kernel simulation tractable (statevector sim cost)

np.random.seed(RANDOM_STATE)


def load_and_prepare_data():
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Subsample for tractable statevector simulation (O(n^2) kernel entries,
    # each requiring a 2^N_QUBITS-dim statevector). Stratified to preserve class balance.
    rng = np.random.default_rng(RANDOM_STATE)
    idx_0 = rng.choice(np.where(y == 0)[0], size=SUBSAMPLE_N // 2, replace=False)
    idx_1 = rng.choice(np.where(y == 1)[0], size=SUBSAMPLE_N // 2, replace=False)
    idx = np.concatenate([idx_0, idx_1])
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    # Standardize, then PCA to N_QUBITS dims to capture max variance in few features
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=N_QUBITS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # Rescale to [0, pi] -- standard range for angle-encoding feature maps
    X_final = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_pca)

    explained_var = pca.explained_variance_ratio_.sum()
    return X_final, y, explained_var, data.feature_names


def evaluate_precomputed_kernel_svm(K_full, y, n_folds=N_FOLDS):
    """Cross-validate an SVM using a precomputed full kernel matrix."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(K_full, y):
        K_train = K_full[np.ix_(train_idx, train_idx)]
        K_test = K_full[np.ix_(test_idx, train_idx)]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = SVC(kernel="precomputed", C=1.0)
        clf.fit(K_train, y_train)
        preds = clf.predict(K_test)

        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds))
    return np.array(accs), np.array(f1s)


def evaluate_classical_model(model, X, y, n_folds=N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": make_scorer(accuracy_score), "f1": make_scorer(f1_score)}
    results = cross_validate(model, X, y, cv=skf, scoring=scoring)
    return results["test_accuracy"], results["test_f1"]


def main():
    print("=" * 70)
    print("QUANTUM KERNEL METHODS FOR MEDICAL CLASSIFICATION")
    print("Breast Cancer Wisconsin (Diagnostic) -- Quantum vs Classical")
    print("=" * 70)

    X, y, explained_var, feature_names = load_and_prepare_data()
    print(f"\nDataset: {X.shape[0]} samples x {X.shape[1]} features "
          f"(PCA-reduced, {explained_var:.1%} variance retained)")
    print(f"Class balance: {np.bincount(y)}")

    results = {}

    # ---- Quantum kernels ----
    for entanglement in ["linear", "full"]:
        print(f"\n[Quantum Kernel | entanglement={entanglement}] Computing kernel matrix...")
        t0 = time.time()
        qk = QuantumKernel(n_qubits=N_QUBITS, reps=2, entanglement=entanglement)
        K = qk.kernel_matrix(X)
        elapsed = time.time() - t0

        alignment = kernel_target_alignment(K, y)
        kl_div, fidelities = circuit_expressibility(qk.feature_map, N_QUBITS)

        accs, f1s = evaluate_precomputed_kernel_svm(K, y)
        key = f"quantum_{entanglement}"
        results[key] = {
            "label": f"Quantum Kernel SVM ({entanglement} entanglement)",
            "cv_accuracy_mean": float(accs.mean()),
            "cv_accuracy_std": float(accs.std()),
            "cv_f1_mean": float(f1s.mean()),
            "cv_f1_std": float(f1s.std()),
            "kernel_target_alignment": alignment,
            "expressibility_kl": kl_div,
            "kernel_computation_time_s": elapsed,
        }
        np.save(f"kernel_matrix_{entanglement}.npy", K)
        print(f"  Kernel computed in {elapsed:.1f}s | "
              f"CV accuracy: {accs.mean():.3f} +/- {accs.std():.3f} | "
              f"Alignment: {alignment:.4f} | Expressibility KL: {kl_div:.4f}")

    # ---- Classical baselines ----
    classical_models = {
        "svm_rbf": ("Classical SVM (RBF kernel)", SVC(kernel="rbf", C=1.0, gamma="scale")),
        "svm_poly": ("Classical SVM (polynomial, deg=3)", SVC(kernel="poly", degree=3, C=1.0)),
        "svm_linear": ("Classical SVM (linear kernel)", SVC(kernel="linear", C=1.0)),
        "neural_net": ("Neural Network (MLP, 2x16)",
                        MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=2000,
                                       random_state=RANDOM_STATE)),
    }

    print("\n[Classical baselines]")
    for key, (label, model) in classical_models.items():
        t0 = time.time()
        accs, f1s = evaluate_classical_model(model, X, y)
        elapsed = time.time() - t0
        results[key] = {
            "label": label,
            "cv_accuracy_mean": float(accs.mean()),
            "cv_accuracy_std": float(accs.std()),
            "cv_f1_mean": float(f1s.mean()),
            "cv_f1_std": float(f1s.std()),
            "kernel_target_alignment": None,
            "expressibility_kl": None,
            "kernel_computation_time_s": elapsed,
        }
        print(f"  {label}: accuracy {accs.mean():.3f} +/- {accs.std():.3f}, "
              f"time {elapsed:.2f}s")

    # ---- Save results ----
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results).T
    df.to_csv("results_summary.csv")

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by CV accuracy)")
    print("=" * 70)
    df_sorted = df.sort_values("cv_accuracy_mean", ascending=False)
    for idx, row in df_sorted.iterrows():
        align_str = f"{row['kernel_target_alignment']:.4f}" if row['kernel_target_alignment'] is not None else "n/a"
        print(f"  {row['label']:<40s} acc={row['cv_accuracy_mean']:.3f}  "
              f"f1={row['cv_f1_mean']:.3f}  alignment={align_str}")

    print("\nResults saved to results.json / results_summary.csv")
    return results, X, y


if __name__ == "__main__":
    main()
