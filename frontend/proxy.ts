import type { NextFetchEvent, NextRequest } from "next/server";
import { getClerkPublishableKey } from "./lib/clerk-config";

const protectedRoutePatterns = [
  "/v2(.*)",
  "/experiments(.*)",
  "/hypotheses(.*)",
  "/validation(.*)",
  "/api/proxy(.*)",
];

const protectedRoutePrefixes = [
  "/v2",
  "/experiments",
  "/hypotheses",
  "/validation",
  "/api/proxy",
];

export default async function proxy(req: NextRequest, event: NextFetchEvent) {
  const publishableKey = getClerkPublishableKey();
  if (!publishableKey || !isProtectedPath(req.nextUrl.pathname)) {
    return;
  }

  const { clerkMiddleware, createRouteMatcher } = await import("@clerk/nextjs/server");
  const protectedRoutes = createRouteMatcher(protectedRoutePatterns);
  const middleware = clerkMiddleware(async (auth, clerkReq) => {
    if (protectedRoutes(clerkReq)) {
      await auth.protect();
    }
  });

  return middleware(req, event);
}

function isProtectedPath(pathname: string): boolean {
  return protectedRoutePrefixes.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`),
  );
}

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/api/(.*)",
  ],
};
