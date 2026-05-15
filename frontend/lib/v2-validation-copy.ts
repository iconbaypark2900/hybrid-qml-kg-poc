export const validationDecisionLabels = ["Keep", "Review", "Reject"] as const;

export const validationSourceVocabulary = [
  "clinical",
  "mechanism",
  "model",
  "baseline",
  "artifact",
] as const;

export const validationTrustLabels = [
  "Clinical support",
  "Mechanism support",
  "Model signal",
  "Baseline comparison",
  "Artifact quality",
] as const;

export type ValidationDecision = (typeof validationDecisionLabels)[number];
export type ValidationSource = (typeof validationSourceVocabulary)[number];
