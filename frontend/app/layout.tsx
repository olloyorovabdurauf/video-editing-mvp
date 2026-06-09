import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Reel Forge — AI Video Editor",
  description: "Turn long-form into viral reels with generative b-roll.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b border-white/5">
          <div className="max-w-5xl mx-auto px-6 py-5 flex items-center justify-between">
            <a href="/" className="text-lg font-semibold tracking-tight">
              <span className="text-accent">▸</span> Reel Forge
            </a>
            <nav className="text-sm text-white/50">
              <a className="hover:text-white transition" href="/api/v1/docs" target="_blank">
                API docs
              </a>
            </nav>
          </div>
        </header>
        <main className="max-w-5xl mx-auto px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
