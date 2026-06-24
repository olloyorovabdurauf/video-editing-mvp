"use client";

const EXAMPLES = [
  "5 mistakes beginners make when starting a business",
  "How I plan a week of content in 30 minutes",
  "Why most diets fail (and what works instead)",
  "The pricing mistake that kills new SaaS products",
  "A 60-second framework for writing better hooks",
];

/** Shown before the first generation: a friendly nudge + tappable example prompts. */
export function EmptyState({ onPick }: { onPick: (topic: string) => void }) {
  return (
    <div className="card border-dashed text-center">
      <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-accent/15 text-accent">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
        </svg>
      </div>
      <p className="text-sm font-medium text-white">Your script will appear here</p>
      <p className="mx-auto mt-1 max-w-sm text-sm text-white/45">
        Describe a topic above, or start from an example:
      </p>
      <div className="mt-4 flex flex-wrap justify-center gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => onPick(ex)}
            className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-white/60
              transition hover:border-accent/40 hover:text-white"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
