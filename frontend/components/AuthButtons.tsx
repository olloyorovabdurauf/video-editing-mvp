"use client";

import { SignInButton, UserButton, useAuth } from "@clerk/nextjs";

/**
 * Header auth UI. Uses the useAuth() hook (stable across Clerk versions) to
 * toggle between a Sign-in button and the user menu, avoiding the
 * version-specific <SignedIn>/<SignedOut> control components.
 */
export function AuthButtons() {
  const { isLoaded, isSignedIn } = useAuth();
  if (!isLoaded) return null;
  return isSignedIn ? (
    <UserButton />
  ) : (
    <SignInButton mode="modal">
      <button className="hover:text-white transition">Sign in</button>
    </SignInButton>
  );
}
