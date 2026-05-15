from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_on_heron import _select_backend_name


class FakeStatus:
    def __init__(self, pending_jobs: int, operational: bool = True) -> None:
        self.pending_jobs = pending_jobs
        self.operational = operational


class FakeBackend:
    def __init__(self, name: str, pending_jobs: int, operational: bool = True) -> None:
        self.name = name
        self._status = FakeStatus(pending_jobs, operational)

    def status(self) -> FakeStatus:
        return self._status


def main() -> None:
    backends = [
        FakeBackend("ibm_fez", pending_jobs=8),
        FakeBackend("ibm_boston", pending_jobs=2),
        FakeBackend("ibm_miami", pending_jobs=1, operational=False),
    ]

    assert _select_backend_name(backends, "ibm_fez") == "ibm_fez"
    assert _select_backend_name(backends, "auto") == "ibm_boston"

    try:
        _select_backend_name(backends, "ibm_torino")
    except ValueError as exc:
        assert "ibm_torino" in str(exc)
        assert "ibm_fez" in str(exc)
    else:
        raise AssertionError("Unavailable backend should raise a helpful error.")


if __name__ == "__main__":
    main()
