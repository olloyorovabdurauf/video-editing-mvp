import "./globals.css";
import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import { AuthBridge } from "@/components/AuthBridge";
import { AuthButtons } from "@/components/AuthButtons";

export const metadata: Metadata = {
  title: "Reel Forge — AI Video Editor",
  description: "Turn long-form into viral reels with generative b-roll.",
};

// Clerk activates only when the key is present, so the build/site never breaks
// if it's unset (e.g. local dev without keys).
const clerkKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b border-white/5">
          <div className="max-w-5xl mx-auto px-6 py-5 flex items-center justify-between">
            <a href="/" className="text-lg font-semibold tracking-tight">
              <span className="text-accent">▸</span> Reel Forge
            </a>
            <nav className="text-sm text-white/50 flex items-center gap-4">
              {clerkKey ? (
                <AuthButtons />
              ) : (
                <a className="hover:text-white transition" href="/api/v1/docs" target="_blank">
                  API docs
                </a>
              )}
            </nav>
          </div>
        </header>
        <main className="max-w-5xl mx-auto px-6 py-10">{children}</main>
      </body>
    </html>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  if (!clerkKey) {
    return <Shell>{children}</Shell>;
  }
  return (
    <ClerkProvider publishableKey={clerkKey}>
      <AuthBridge />
      <Shell>{children}</Shell>
    </ClerkProvider>
  );
}
