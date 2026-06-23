import { clerkMiddleware } from "@clerk/nextjs/server";
import { NextResponse, type NextRequest } from "next/server";
import type { NextFetchEvent } from "next/server";

// Attaches Clerk auth context to page requests. We deliberately EXCLUDE
// /api and /storage — those are proxied to the FastAPI backend, which verifies
// the JWT itself (sent in the Authorization header by the client fetch).
//
// SAFETY: clerkMiddleware() throws at runtime if CLERK_SECRET_KEY is absent.
// We only engage it when Clerk is fully configured; otherwise the site serves
// normally with auth inactive. This makes enabling Clerk a pure env-var flip —
// adding the secret + redeploying turns it on, with no broken-window window.
const clerkConfigured =
  !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && !!process.env.CLERK_SECRET_KEY;

const clerkHandler = clerkMiddleware();

export default function middleware(req: NextRequest, ev: NextFetchEvent) {
  if (!clerkConfigured) return NextResponse.next();
  return clerkHandler(req, ev);
}

export const config = {
  matcher: ["/((?!_next|api|storage|.*\\..*).*)"],
};
