export type AppRole = "researcher" | "reviewer" | "admin";
export type DatasetScope =
  | "hetionet:read"
  | "molecule:read"
  | "quantum:read"
  | "quantum:write"
  | "sessions:write"
  | "sessions:export"
  | "experiments:launch"
  | "audit:read";

const roleScopes: Record<AppRole, DatasetScope[]> = {
  researcher: [
    "hetionet:read",
    "molecule:read",
    "quantum:read",
    "quantum:write",
    "sessions:write",
  ],
  reviewer: [
    "hetionet:read",
    "molecule:read",
    "quantum:read",
    "quantum:write",
    "sessions:write",
    "sessions:export",
  ],
  admin: [
    "hetionet:read",
    "molecule:read",
    "quantum:read",
    "quantum:write",
    "sessions:write",
    "sessions:export",
    "experiments:launch",
    "audit:read",
  ],
};

export function scopesForRole(role: AppRole): DatasetScope[] {
  return roleScopes[role];
}

export function hasScope(scopes: readonly DatasetScope[], scope: DatasetScope): boolean {
  return scopes.includes(scope);
}

export function roleForEmail(
  email: string | null | undefined,
  adminEmails = process.env.DEV_ADMIN_EMAILS ?? "",
  reviewerEmails = process.env.DEV_REVIEWER_EMAILS ?? "",
): AppRole {
  const normalized = (email ?? "").trim().toLowerCase();
  if (normalized && parseEmailList(adminEmails).has(normalized)) return "admin";
  if (normalized && parseEmailList(reviewerEmails).has(normalized)) return "reviewer";
  return "researcher";
}

export function getRequiredScopeForApiPath(
  method: string,
  path: string,
): DatasetScope | null {
  const normalizedMethod = method.toUpperCase();
  const normalizedPath = normalizePath(path);

  if (normalizedPath.startsWith("/audit-events")) return "audit:read";
  if (normalizedMethod === "POST" && normalizedPath === "/jobs/pipeline") {
    return "experiments:launch";
  }
  if (
    normalizedPath.startsWith("/research-sessions") &&
    normalizedPath.endsWith("/export")
  ) {
    return "sessions:export";
  }
  if (
    normalizedPath.startsWith("/research-sessions") &&
    ["POST", "PATCH"].includes(normalizedMethod)
  ) {
    return "sessions:write";
  }
  if (normalizedPath === "/config/ibm-quantum" && normalizedMethod === "POST") {
    return "quantum:write";
  }
  if (normalizedPath.startsWith("/config/ibm-quantum")) {
    return "quantum:read";
  }
  if (normalizedPath.startsWith("/viz/molecule")) return "molecule:read";
  if (normalizedPath.startsWith("/viz/circuit") || normalizedPath.startsWith("/quantum")) {
    return "quantum:read";
  }
  if (normalizedPath.startsWith("/viz") || normalizedPath.startsWith("/kg")) {
    return "hetionet:read";
  }
  return null;
}

function normalizePath(path: string): string {
  return path.startsWith("/") ? path : `/${path}`;
}

function parseEmailList(value: string): Set<string> {
  return new Set(
    value
      .split(",")
      .map((item) => item.trim().toLowerCase())
      .filter(Boolean),
  );
}
