import { auth, currentUser } from "@clerk/nextjs/server";
import { NextRequest, NextResponse } from "next/server";
import {
  getRequiredScopeForApiPath,
  hasScope,
  roleForEmail,
  scopesForRole,
} from "@/lib/authz";

const apiOrigin = (
  process.env.API_ORIGIN ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

type RouteContext = {
  params: Promise<{ path?: string[] }>;
};

export async function GET(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function POST(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function PUT(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  return proxyRequest(request, context);
}

async function proxyRequest(request: NextRequest, context: RouteContext) {
  const { path = [] } = await context.params;
  const targetPath = `/${path.join("/")}`;
  const access = await resolveAccess(request.method, targetPath);

  if (!access.ok) {
    return NextResponse.json(
      { detail: access.message },
      { status: access.status },
    );
  }

  const url = new URL(request.url);
  const target = `${apiOrigin}${targetPath}${url.search}`;
  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.set("x-qgg-user-id", access.userId);
  headers.set("x-qgg-user-email", access.email ?? "");
  headers.set("x-qgg-role", access.role);
  headers.set("x-qgg-scopes", access.scopes.join(","));
  if (process.env.QGG_INTERNAL_API_SECRET) {
    headers.set("x-qgg-internal-signature", process.env.QGG_INTERNAL_API_SECRET);
  }

  const body = ["GET", "HEAD"].includes(request.method)
    ? undefined
    : await request.arrayBuffer();

  const response = await fetch(target, {
    method: request.method,
    headers,
    body,
    cache: "no-store",
  });

  return new NextResponse(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: filterResponseHeaders(response.headers),
  });
}

async function resolveAccess(method: string, path: string): Promise<
  | {
      ok: true;
      userId: string;
      email: string | null;
      role: ReturnType<typeof roleForEmail>;
      scopes: ReturnType<typeof scopesForRole>;
    }
  | { ok: false; status: number; message: string }
> {
  const requiredScope = getRequiredScopeForApiPath(method, path);
  const clerkEnabled = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);

  if (!clerkEnabled) {
    const role = roleForEmail(null);
    const scopes = scopesForRole(role);
    if (requiredScope && !hasScope(scopes, requiredScope)) {
      return {
        ok: false,
        status: 403,
        message: `Missing required scope: ${requiredScope}`,
      };
    }
    return {
      ok: true,
      userId: "dev-user",
      email: null,
      role,
      scopes,
    };
  }

  const { userId } = await auth();
  if (!userId) {
    return { ok: false, status: 401, message: "Unauthorized" };
  }

  const user = await currentUser();
  const email = user?.primaryEmailAddress?.emailAddress ?? null;
  const metadataRole = user?.publicMetadata?.role;
  const role =
    metadataRole === "admin" || metadataRole === "reviewer" || metadataRole === "researcher"
      ? metadataRole
      : roleForEmail(email);
  const scopes = scopesForRole(role);

  if (requiredScope && !hasScope(scopes, requiredScope)) {
    return {
      ok: false,
      status: 403,
      message: `Missing required scope: ${requiredScope}`,
    };
  }

  return {
    ok: true,
    userId,
    email,
    role,
    scopes,
  };
}

function filterResponseHeaders(headers: Headers) {
  const out = new Headers(headers);
  out.delete("content-encoding");
  out.delete("content-length");
  out.delete("transfer-encoding");
  return out;
}
