"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
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
    return <div className="text-red-400">Error: {error}</div>;
  }
  if (!job) {
    return <div className="text-white/50">Loading job…</div>;
  }

  return (
    <div className="space-y-8">
      <JobProgress job={job} />

      {job.status === "failed" && (
        <div className="card border-red-500/30 bg-red-500/5">
          <h3 className="text-red-400 font-medium mb-2">Job failed</h3>
          <p className="text-sm text-white/70">{job.message || "Unknown error"}</p>
        </div>
      )}

      {job.artifacts.length > 0 && (
        <section>
          <h2 className="text-xl font-semibold mb-4">Generated reels</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {job.artifacts.map((reel, i) => (
              <ReelCard key={i} reel={reel} index={i} />
            ))}
          </div>
        </section>
      )}

      {job.status !== "succeeded" && job.status !== "failed" && job.artifacts.length === 0 && (
        <div className="text-center text-white/40 text-sm py-10">
          Reels will appear here as they finish rendering.
        </div>
      )}
    </div>
  );
}
