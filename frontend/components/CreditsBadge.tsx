"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { getBalance } from "@/lib/api";

/**
 * Header credits indicator. Only renders for signed-in users; silently hides if
 * the balance endpoint isn't reachable (e.g. auth disabled in local dev).
 */
export function CreditsBadge() {
  const { isSignedIn, isLoaded } = useAuth();
  const [balance, setBalance] = useState<number | null>(null);

  useEffect(() => {
    if (!isLoaded || !isSignedIn) return;
    let alive = true;
    getBalance()
      .then((b) => alive && setBalance(b.balance))
      .catch(() => alive && setBalance(null));
    return () => {
      alive = false;
    };
  }, [isLoaded, isSignedIn]);

  if (!isSignedIn || balance === null) return null;

  return (
    <span
      title="Available credits"
      className="inline-flex items-center gap-1.5 rounded-full border border-white/10
        bg-white/5 px-3 py-1.5 text-xs font-medium text-white/80"
    >
      <span className="text-accent">◆</span>
      {balance.toLocaleString()}
      <span className="hidden text-white/40 sm:inline">credits</span>
    </span>
  );
}
