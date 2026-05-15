import { parseV2Session } from "./v2-data";
import { loadV2ExperimentEvidence, loadV2VisualEvidence } from "./v2-live-data";

async function assertEvidenceLoadersCompile() {
  const session = parseV2Session({
    entity: "Atherosclerosis",
    runMode: "Hybrid",
    candidate: "Atherosclerosis",
  });

  const experiment = await loadV2ExperimentEvidence(session);
  const visual = await loadV2VisualEvidence(session);

  if (experiment.status.source === "error") {
    throw new Error("Experiment evidence loader must expose an error state.");
  }

  if (visual.molecule.source === "fallback" && visual.molecule.atoms.length > 0) {
    throw new Error("Fallback molecule evidence should not expose live atoms.");
  }
}

void assertEvidenceLoadersCompile;
