import type { Metadata } from "next";
import { ScriptStudio } from "@/components/studio/ScriptStudio";

export const metadata: Metadata = {
  title: "Script Studio — Reel Forge",
  description: "Generate high-retention short-form video scripts in seconds.",
};

// Authed, interactive page — render per request (and so local builds without a
// Clerk key don't try to prerender Clerk hooks).
export const dynamic = "force-dynamic";

export default function StudioPage() {
  return <ScriptStudio />;
}
