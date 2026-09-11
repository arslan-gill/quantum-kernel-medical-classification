"""
Quantum Kernel Methods for Medical Classification
===================================================
Implements a quantum feature-map kernel (PennyLane) for binary classification,
designed to be benchmarked against classical kernels (RBF, polynomial, linear)
and a classical neural network baseline.

Core research question:
    Does a quantum kernel built from an entangling feature map capture
    structure in a small-sample, moderate-dimensional medical dataset
    that classical kernels miss -- and if so, can we characterize why
    (via kernel-target alignment and circuit expressibility)?
"""

import numpy as np
import pennylane as qml
from itertools import combinations


def make_feature_map_circuit(n_qubits, reps=2, entanglement="linear"):
    """
    Builds a ZZ-FeatureMap-style quantum feature map (Havlicek et al., 2019 style).
    Each repetition: Hadamard layer + data-encoded RZ rotations + entangling ZZ layer.

    Returns a PennyLane QNode that outputs the statevector for a given input x.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def entangling_pairs():
        if entanglement == "linear":
            return [(i, i + 1) for i in range(n_qubits - 1)]
        elif entanglement == "full":
            return list(combinations(range(n_qubits), 2))
        else:
            raise ValueError("entanglement must be 'linear' or 'full'")

    @qml.qnode(dev)
    def feature_map(x):
        pairs = entangling_pairs()
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * x[i], wires=i)
            for (i, j) in pairs:
                qml.CNOT(wires=[i, j])
                qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=j)
                qml.CNOT(wires=[i, j])
        return qml.state()

    return feature_map, dev


class QuantumKernel:
    """
    Computes a fidelity-based quantum kernel:
        k(x, y) = |<phi(x)|phi(y)>|^2
    using statevector simulation (exact, noise-free).
    """

    def __init__(self, n_qubits, reps=2, entanglement="linear"):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.feature_map, self.dev = make_feature_map_circuit(n_qubits, reps, entanglement)
        self._state_cache = {}

    def _state(self, x):
        key = tuple(np.round(x, 10))
        if key not in self._state_cache:
            self._state_cache[key] = np.array(self.feature_map(x))
        return self._state_cache[key]

    def kernel_value(self, x, y):
        sx = self._state(x)
        sy = self._state(y)
        overlap = np.vdot(sx, sy)
        return float(np.abs(overlap) ** 2)

    def kernel_matrix(self, X, Y=None):
        symmetric = Y is None
        if symmetric:
            Y = X
        n, m = len(X), len(Y)
        K = np.zeros((n, m))
        if symmetric:
            for i in range(n):
                for j in range(i, n):
                    val = self.kernel_value(X[i], Y[j])
                    K[i, j] = val
                    K[j, i] = val
        else:
            for i in range(n):
                for j in range(m):
                    K[i, j] = self.kernel_value(X[i], Y[j])
        return K

    def clear_cache(self):
        self._state_cache = {}


def kernel_target_alignment(K, y):
    """
    Computes kernel-target alignment (Cristianini et al., 2001):
        A(K, yy^T) = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    A higher value means the kernel's similarity structure agrees more
    with the label structure -- a proxy for how "useful" the kernel is
    for this classification task, independent of any downstream classifier.
    """
    y = np.array(y).astype(float)
    y = 2 * y - 1  # map {0,1} -> {-1,+1}
    Y = np.outer(y, y)
    numerator = np.sum(K * Y)
    denominator = np.linalg.norm(K, "fro") * np.linalg.norm(Y, "fro")
    return float(numerator / denominator)


def circuit_expressibility(feature_map, n_qubits, n_samples=200, n_bins=75, seed=42):
    """
    Approximates circuit expressibility (Sim, Johnson & Aspuru-Guzik, 2019) via
    KL divergence between the empirical fidelity distribution of randomly sampled
    input pairs and the theoretical fidelity distribution of Haar-random states.

    Lower KL divergence => more expressible (closer to Haar-random coverage of
    Hilbert space) => richer feature map, but also higher risk of barren plateaus.
    """
    rng = np.random.default_rng(seed)
    fidelities = []
    for _ in range(n_samples):
        x = rng.uniform(0, 2 * np.pi, size=n_qubits)
        y = rng.uniform(0, 2 * np.pi, size=n_qubits)
        sx = np.array(feature_map(x))
        sy = np.array(feature_map(y))
        fidelities.append(np.abs(np.vdot(sx, sy)) ** 2)
    fidelities = np.array(fidelities)

    # Empirical distribution
    hist, edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    hist = np.clip(hist, 1e-12, None)

    # Haar-random fidelity distribution for N-dim Hilbert space: P(F) = (N-1)(1-F)^(N-2)
    N = 2 ** n_qubits
    centers = 0.5 * (edges[:-1] + edges[1:])
    haar_pdf = (N - 1) * (1 - centers) ** (N - 2)
    haar_pdf = haar_pdf / haar_pdf.sum()
    haar_pdf = np.clip(haar_pdf, 1e-12, None)

    kl_div = float(np.sum(hist * np.log(hist / haar_pdf)))
    return kl_div, fidelities
