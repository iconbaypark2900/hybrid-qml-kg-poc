import {
  getRequiredScopeForApiPath,
  hasScope,
  roleForEmail,
  scopesForRole,
} from "./authz";

if (!hasScope(scopesForRole("reviewer"), "sessions:export")) {
  throw new Error("Reviewers must be allowed to export evidence packets.");
}

if (hasScope(scopesForRole("researcher"), "sessions:export")) {
  throw new Error("Researchers should not export evidence packets by default.");
}

if (getRequiredScopeForApiPath("POST", "/jobs/pipeline") !== "experiments:launch") {
  throw new Error("Pipeline launch must require experiment-launch scope.");
}

if (getRequiredScopeForApiPath("POST", "/config/ibm-quantum") !== "quantum:write") {
  throw new Error("Saving IBM Quantum credentials must require quantum-write scope.");
}

if (getRequiredScopeForApiPath("GET", "/config/ibm-quantum") !== "quantum:read") {
  throw new Error("Reading IBM Quantum metadata must require quantum-read scope.");
}

if (roleForEmail("admin@example.com", "admin@example.com", "") !== "admin") {
  throw new Error("DEV_ADMIN_EMAILS should promote local admin users.");
}
