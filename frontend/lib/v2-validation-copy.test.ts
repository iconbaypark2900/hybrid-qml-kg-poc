import {
  validationDecisionLabels,
  validationSourceVocabulary,
  validationTrustLabels,
} from "./v2-validation-copy";

const expectedDecisions = ["Keep", "Review", "Reject"];
const expectedSources: string[] = ["clinical", "mechanism", "model", "baseline", "artifact"];
const expectedTrustLabels = [
  "Clinical support",
  "Mechanism support",
  "Model signal",
  "Baseline comparison",
  "Artifact quality",
];

if (validationDecisionLabels.join("|") !== expectedDecisions.join("|")) {
  throw new Error("Validation decision labels drifted from Keep/Review/Reject.");
}

const sourceVocabulary: readonly string[] = validationSourceVocabulary;
for (const source of expectedSources) {
  if (!sourceVocabulary.includes(source)) {
    throw new Error(`Missing validation source vocabulary: ${source}`);
  }
}

if (validationTrustLabels.join("|") !== expectedTrustLabels.join("|")) {
  throw new Error("Validation trust labels drifted from the trust scorecard contract.");
}
