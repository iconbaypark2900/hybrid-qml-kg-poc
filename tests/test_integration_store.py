from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from middleware.integration_store import IntegrationStore


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = IntegrationStore(Path(tmp) / "integration_state.db")

        saved = store.save_ibm_quantum_credentials(
            tenant_id="tenant-a",
            token="ibm-token-a",
            instance_crn="crn:v1:bluemix:public:quantum-computing:::tenant-a",
        )

        assert saved["configured"] is True
        assert saved["tenant_id"] == "tenant-a"
        assert saved["provider"] == "ibm_quantum"
        assert saved["instance_crn"].endswith("tenant-a")
        assert saved["token_preview"] == "ibm...n-a"

        metadata = store.get_ibm_quantum_metadata("tenant-a")
        assert metadata is not None
        assert metadata["configured"] is True
        assert metadata["token_preview"] == "ibm...n-a"

        credentials = store.get_ibm_quantum_credentials("tenant-a")
        assert credentials == {
            "token": "ibm-token-a",
            "instance_crn": "crn:v1:bluemix:public:quantum-computing:::tenant-a",
            "channel": "ibm_quantum_platform",
        }

        assert store.get_ibm_quantum_metadata("tenant-b") is None
        assert store.get_ibm_quantum_credentials("tenant-b") is None


if __name__ == "__main__":
    main()
