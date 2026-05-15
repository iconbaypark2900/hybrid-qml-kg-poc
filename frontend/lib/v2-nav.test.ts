import { v2NavItems } from "./v2-nav";

const labels: string[] = v2NavItems.map((item) => item.label);
const expected = ["Initialize", "Experiment", "Validate", "Visualize", "Operations"];

if (labels.join("|") !== expected.join("|")) {
  throw new Error(`Expected canonical v2 nav labels ${expected.join(", ")}, got ${labels.join(", ")}`);
}

if (labels.some((label) => label === "Start" || label === "Ops")) {
  throw new Error("Legacy v2 labels should not be exposed in the canonical shell.");
}

if (v2NavItems.find((item) => item.label === "Initialize")?.href !== "/v2/start") {
  throw new Error("Initialize should keep the /v2/start route for link compatibility.");
}
