import {
  experimentMetricLabels,
  experimentPrimaryActions,
  experimentSourceLabels,
  experimentSourceVocabulary,
} from "./v2-experiment-copy";

const expectedMetrics = [
  "Starting point",
  "Candidate under review",
  "Evidence source",
  "Run path",
];

if (experimentMetricLabels.join("|") !== expectedMetrics.join("|")) {
  throw new Error("Experiment metric labels drifted from the cockpit contract.");
}

for (const label of ["API status", "Latest run", "Candidate source", "Job source"]) {
  if (!experimentSourceLabels.includes(label as (typeof experimentSourceLabels)[number])) {
    throw new Error(`Missing Experiment source label: ${label}`);
  }
}

for (const source of ["live", "fallback", "empty", "error"]) {
  if (!experimentSourceVocabulary.includes(source as (typeof experimentSourceVocabulary)[number])) {
    throw new Error(`Missing Experiment source vocabulary: ${source}`);
  }
}

if (!experimentPrimaryActions.includes("Send candidate to Validate")) {
  throw new Error("Experiment must keep Validate as the primary candidate action.");
}
