from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import qiskit.transpiler.preset_passmanagers as preset_passmanagers
import quantum_layer.quantum_executor as qe


class FakeBackend:
    pass


class FakeService:
    def backend(self, name):
        assert name == "ibm_kingston"
        return FakeBackend()


class FakeSession:
    def __init__(self, backend):
        self.backend = backend


class FakeSampler:
    def __init__(self, mode=None, options=None, **kwargs):
        if "session" in kwargs:
            raise TypeError("unexpected session keyword")
        if not isinstance(options, FakeSamplerOptions):
            raise TypeError("invalid options type")
        self.mode = mode
        self.options = options


class FakeSamplerOptions:
    def __init__(self):
        self.default_shots = None
        self.max_execution_time = None


class FakeCircuit:
    def __init__(self):
        self.global_phase = "parameterized-phase"


class FakePassManager:
    def run(self, circuit):
        circuit.global_phase = "parameterized-phase"
        return circuit


def main() -> None:
    old_session = qe.Session
    old_sampler = qe.RuntimeSampler
    old_sampler_options = getattr(qe, "SamplerOptions", None)
    old_generate = preset_passmanagers.generate_preset_pass_manager
    try:
        qe.Session = FakeSession
        qe.RuntimeSampler = FakeSampler
        qe.SamplerOptions = FakeSamplerOptions
        preset_passmanagers.generate_preset_pass_manager = lambda **kwargs: FakePassManager()
        executor = qe.QuantumExecutor.__new__(qe.QuantumExecutor)
        executor.service = FakeService()
        executor.session = None
        executor.config = {
            "quantum": {
                "heron": {
                    "backend": "ibm_kingston",
                    "shots": 100,
                    "max_runtime_minutes": 30,
                }
            }
        }

        sampler, backend_name = executor._get_heron_sampler()

        assert backend_name == "ibm_kingston"
        assert isinstance(sampler.mode, FakeSession)
        assert sampler.mode.backend.__class__ is FakeBackend
        assert sampler.options.default_shots == 100
        assert getattr(sampler, "_qgg_backend", None).__class__ is FakeBackend
        circuit = sampler._qgg_pass_manager.run(FakeCircuit())
        assert circuit.global_phase == 0
    finally:
        qe.Session = old_session
        qe.RuntimeSampler = old_sampler
        qe.SamplerOptions = old_sampler_options
        preset_passmanagers.generate_preset_pass_manager = old_generate


if __name__ == "__main__":
    main()
