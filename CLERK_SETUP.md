# Connecting Clerk authentication (backend is already ready)

The backend already verifies Clerk JWTs (`app/core/auth.py`, `AUTH_MODE=clerk`)
and enforces ownership on every route. Three steps remain — all on your side
because they need a Clerk account.

## 1. Create the Clerk app (5 min)
1. clerk.com → create an application → enable Email + Google.
2. Copy from **API Keys**:
   - `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` (frontend)
   - the **JWKS URL**: `https://<your-app>.clerk.accounts.dev/.well-known/jwks.json`

## 2. Turn the backend on (1 command)
```bash
flyctl secrets set AUTH_MODE=clerk \
  CLERK_JWKS_URL="https://<your-app>.clerk.accounts.dev/.well-known/jwks.json" \
  APP_ENV=production CORS_ORIGINS="https://yourdomain.com" \
  -a reelforge-mvp-x7k2
```
`APP_ENV=production` makes auth mandatory (the app refuses to start with
`AUTH_MODE=none`), hides `/docs`, and locks CORS.

## 3. Wire the frontend (apply this patch — ~15 min)

```bash
cd frontend && npm install @clerk/nextjs
```

`frontend/.env.local` (and Vercel env):
```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_live_...
CLERK_SECRET_KEY=sk_live_...
```

`frontend/middleware.ts` (new):
```ts
import { clerkMiddleware } from "@clerk/nextjs/server";
export default clerkMiddleware();
export const config = { matcher: ["/((?!_next|.*\\..*).*)", "/(api|trpc)(.*)"] };
```

`frontend/app/layout.tsx` — wrap the tree:
```tsx
import { ClerkProvider } from "@clerk/nextjs";
// ...
return (
  <ClerkProvider>
    <html lang="en"><body>{children}</body></html>
  </ClerkProvider>
);
```

Send the token on every API call — `frontend/lib/api.ts` already has
`setTokenGetter()`. In a client component near the root:
```tsx
"use client";
import { useAuth } from "@clerk/nextjs";
import { useEffect } from "react";
import { setTokenGetter } from "@/lib/api";
export function AuthBridge() {
  const { getToken } = useAuth();
  useEffect(() => { setTokenGetter(() => getToken()); }, [getToken]);
  return null;
}
```
Render `<AuthBridge/>` inside `ClerkProvider`. Now every `createReel`/`getJob`/
`/uploads` call carries the verified JWT; the backend extracts `sub` and enforces
ownership.

## 4. Verify
- Logged out → API returns `401`.
- Logged in → jobs are tied to your Clerk `sub`; you cannot read another user's job (`404`).
