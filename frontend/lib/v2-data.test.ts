import {
  buildV2Params,
  getV2CandidateForEntity,
  parseV2Session,
} from "./v2-data";

const losartanSession = parseV2Session({
  entity: "Losartan",
  runMode: "Quantum hardware",
  candidate: "Atherosclerosis",
});

if (losartanSession.selectedEntity.name !== "Losartan") {
  throw new Error("Expected Losartan to round-trip from URL params.");
}

if (losartanSession.selectedCandidate.disease !== "Atherosclerosis") {
  throw new Error("Expected selected candidate to round-trip from URL params.");
}

const losartanCandidate = getV2CandidateForEntity("Atherosclerosis");
if (losartanCandidate.candidate !== "Losartan") {
  throw new Error("Expected paper-aligned Atherosclerosis candidate.");
}

const params = buildV2Params(losartanSession);
params satisfies string;
