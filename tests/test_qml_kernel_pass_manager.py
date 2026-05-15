from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import quantum_layer.qml_model as qm


class FakeSampler:
    pass


class FakeFidelity:
    def __init__(self, sampler, pass_manager=None):
        self.sampler = sampler
        self.pass_manager = pass_manager


class FakeKernel:
    def __init__(self, feature_map, fidelity):
        self.feature_map = feature_map
        self.fidelity = fidelity


def main() -> None:
    old_fidelity = qm.ComputeUncompute
    old_kernel = qm.FidelityQuantumKernel
    try:
        qm.ComputeUncompute = FakeFidelity
        qm.FidelityQuantumKernel = FakeKernel

        sampler = FakeSampler()
        sampler._qgg_pass_manager = object()

        predictor = qm.QMLLinkPredictor(model_type="QSVC", num_qubits=2)
        kernel = predictor._prepare_quantum_kernel(sampler=sampler, exec_mode="heron")

        assert kernel.fidelity.sampler is sampler
        assert kernel.fidelity.pass_manager is sampler._qgg_pass_manager
    finally:
        qm.ComputeUncompute = old_fidelity
        qm.FidelityQuantumKernel = old_kernel


if __name__ == "__main__":
    main()
