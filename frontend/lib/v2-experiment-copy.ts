export const experimentMetricLabels = [
  "Starting point",
  "Candidate under review",
  "Evidence source",
  "Run path",
] as const;

export const experimentSourceLabels = [
  "API status",
  "Latest run",
  "Candidate source",
  "Job source",
] as const;

export const experimentPrimaryActions = [
  "Send candidate to Validate",
  "Visualize evidence",
  "Launch or monitor run",
] as const;

export const experimentSourceVocabulary = [
  "live",
  "fallback",
  "empty",
  "error",
] as const;
