"use client";

import { useState } from "react";

/** Reusable copy-to-clipboard control with a transient "Copied" confirmation. */
export function CopyButton({
  text,
  label = "Copy",
  className = "",
}: {
  text: string;
  label?: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked — no-op, button just doesn't confirm */
    }
  }

  return (
    <button
      type="button"
      onClick={copy}
      className={`inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5
        px-3 py-2 text-xs font-medium text-white/70 transition hover:border-white/20
        hover:text-white active:scale-[0.97] ${className}`}
    >
      {copied ? (
        <>
          <span className="text-accent">✓</span> Copied
        </>
      ) : (
        <>
          <CopyIcon /> {label}
        </>
      )}
    </button>
  );
}

function CopyIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}
