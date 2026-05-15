from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import quantum_layer.qml_model as qm
import quantum_layer.quantum_executor as qexec


class FakeExecutor:
    def __init__(self, config_path):
        self.config_path = config_path
        self.closed = False

    def get_sampler(self):
        return object(), "heron"

    def close_session(self):
        self.closed = True


class FakeKernel:
    pass


class FakeQSVC:
    def __init__(self, quantum_kernel):
        self.quantum_kernel = quantum_kernel
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def decision_function(self, X):
        return np.zeros(X.shape[0])


def main() -> None:
    old_executor = qexec.QuantumExecutor
    old_kernel = qm.QMLLinkPredictor._prepare_quantum_kernel
    old_qsvc = qm.QSVC
    try:
        qexec.QuantumExecutor = FakeExecutor
        qm.QMLLinkPredictor._prepare_quantum_kernel = lambda self, sampler, exec_mode: FakeKernel()
        qm.QSVC = FakeQSVC

        model = qm.QMLLinkPredictor(model_type="QSVC", num_qubits=2)
        model.fit(np.zeros((2, 2)), np.array([0, 1]))

        assert model._quantum_executor is not None
        assert model._quantum_executor.closed is False

        probs = model.predict_proba(np.zeros((1, 2)))

        assert probs.shape == (1, 2)
        assert model._quantum_executor.closed is False
    finally:
        qexec.QuantumExecutor = old_executor
        qm.QMLLinkPredictor._prepare_quantum_kernel = old_kernel
        qm.QSVC = old_qsvc


if __name__ == "__main__":
    main()
