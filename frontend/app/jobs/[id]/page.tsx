"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getJob, type JobResponse } from "@/lib/api";
import { JobProgress } from "@/components/JobProgress";
import { ReelCard } from "@/components/ReelCard";

const POLL_MS = 2_500;

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function tick() {
      try {
        const j = await getJob(id as string);
        if (!alive) return;
        setJob(j);
        if (j.status !== "succeeded" && j.status !== "failed") {
          timer = setTimeout(tick, POLL_MS);
        }
      } catch (e: any) {
        if (!alive) return;
        setError(e.message || "Polling failed");
        timer = setTimeout(tick, POLL_MS * 2);
      }
    }

    tick();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [id]);

  if (error && !job) {
    return (
      <div className="mx-auto max-w-2xl">
        <div className="card border-rose-500/30 bg-rose-500/5 text-center text-sm text-rose-300">
          {error}
        </div>
      </div>
    );
  }
  if (!job) {
    return <div className="mx-auto max-w-2xl text-center text-sm text-white/40">Loading…</div>;
  }

  const done = job.status === "succeeded";

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-white sm:text-2xl">
          {done ? "Your clips" : "Creating your clips"}
        </h1>
        <Link
          href="/"
          className="rounded-lg border border-white/10 px-3 py-2 text-xs font-medium text-white/70
            transition hover:border-white/20 hover:text-white"
        >
          ＋ New video
        </Link>
      </div>

      {/* Show progress until clips arrive */}
      {(!done || job.artifacts.length === 0) && (
        <div className="mx-auto max-w-2xl">
          <JobProgress job={job} />
        </div>
      )}

      {job.status === "failed" && (
        <div className="mx-auto max-w-2xl">
          <div className="card border-rose-500/30 bg-rose-500/5">
            <p className="text-sm font-medium text-rose-300">
              {job.message || "This video couldn't be processed. Try another link."}
            </p>
            <Link href="/" className="mt-3 inline-block text-xs text-white/70 underline hover:text-white">
              Try another video
            </Link>
          </div>
        </div>
      )}

      {job.artifacts.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {job.artifacts.map((reel, i) => (
            <ReelCard key={i} reel={reel} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
