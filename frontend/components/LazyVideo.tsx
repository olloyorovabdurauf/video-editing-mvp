"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Poster-first, lazy-loaded video.
 *
 * Why: rendering N <video> elements that each fetch metadata on mount makes a
 * grid of clips slow and janky. Instead we show the poster image immediately
 * (instant, ~30KB) and only mount the real <video> — which then streams via
 * faststart + HTTP range — when the user clicks play. Offscreen cards don't
 * fetch anything until they scroll near the viewport.
 */
export function LazyVideo({ src, poster }: { src: string; poster?: string | null }) {
  const ref = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);   // scrolled close to viewport
  const [active, setActive] = useState(false); // user pressed play → load full video

  useEffect(() => {
    const el = ref.current;
    if (!el || near) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setNear(true);
          io.disconnect();
        }
      },
      { rootMargin: "300px" },   // warm up just before it's visible
    );
    io.observe(el);
    return () => io.disconnect();
  }, [near]);

  return (
    <div ref={ref} className="relative h-full w-full">
      {active ? (
        <video
          src={src}
          controls
          autoPlay
          preload="metadata"
          poster={poster || undefined}
          className="h-full w-full object-contain"
        />
      ) : (
        <button
          type="button"
          onClick={() => setActive(true)}
          aria-label="Play clip"
          className="group/vid relative block h-full w-full"
        >
          {poster ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={poster} alt="" loading="lazy" className="h-full w-full object-contain" />
          ) : near ? (
            // No poster yet (older clip) → metadata-only video shows its first frame.
            <video src={src} preload="metadata" muted playsInline className="h-full w-full object-contain" />
          ) : (
            <div className="h-full w-full bg-black/40" />
          )}
          {/* Play affordance */}
          <span className="absolute inset-0 grid place-items-center">
            <span className="grid h-14 w-14 place-items-center rounded-full bg-black/50 text-white backdrop-blur transition group-hover/vid:scale-110 group-hover/vid:bg-black/70">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                <path d="M8 5v14l11-7z" />
              </svg>
            </span>
          </span>
        </button>
      )}
    </div>
  );
}
