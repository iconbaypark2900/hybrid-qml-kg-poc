from pathlib import Path
import tempfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from middleware.research_store import ResearchStore


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ResearchStore(Path(tmp) / "research_state.db")
        created = store.create_research_session(
            title="Losartan evidence review",
            reviewer_name="Reviewer One",
            reviewer_email="reviewer@example.com",
            selected_entity={"name": "Atherosclerosis", "type": "Disease"},
            selected_candidate={"candidate": "Losartan", "disease": "Atherosclerosis"},
            run_mode="Hybrid",
            score_threshold="0.65",
            mechanism_weight="High",
            decision="Review",
            notes="Initial review",
            evidence_state={"source": "fallback"},
            provenance=[{"endpoint": "/viz/run-predictions", "source_kind": "fallback"}],
            hypothesis_id=None,
        )

        assert created["id"].startswith("RS-")
        assert store.get_research_session(created["id"])["reviewer_email"] == "reviewer@example.com"

        updated = store.update_research_session(
            created["id"],
            {"decision": "Keep", "notes": "Validated with evidence"},
        )
        assert updated["decision"] == "Keep"

        packet = store.export_research_session(created["id"])
        assert packet["session"]["id"] == created["id"]
        assert packet["session"]["exported_at"] is not None
        assert packet["evidence_packet_version"] == 1


if __name__ == "__main__":
    main()
