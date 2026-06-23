"use client";

import { useAuth } from "@clerk/nextjs";
import { useEffect } from "react";
import { setTokenGetter } from "@/lib/api";

/**
 * Bridges Clerk's session token into our API client so every backend call
 * (createReel / getJob / uploads) carries the verified JWT. lib/api.ts already
 * exposes setTokenGetter(); this just feeds it Clerk's getToken().
 */
export function AuthBridge() {
  const { getToken } = useAuth();
  useEffect(() => {
    setTokenGetter(() => getToken());
  }, [getToken]);
  return null;
}
