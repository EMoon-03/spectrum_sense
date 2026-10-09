import type { Metadata } from "next";
import SiteHeader from "@/components/SiteHeader";
import "./globals.css";

export const metadata: Metadata = {
  title: "SpectrumSense | Interactive signal lab",
  description:
    "Simulate noise, BPSK, and QPSK I/Q windows, classify your own SDR recordings with a synthetic baseline that says \"uncertain\" when it should, and read what the model measured about itself.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col bg-paper text-ink">
        <SiteHeader />
        <div className="flex-1">{children}</div>

        <footer className="mt-20 border-t border-rule bg-surface">
          <div className="mx-auto max-w-6xl px-6 py-8">
            <p className="max-w-2xl text-sm leading-relaxed text-ink-muted">
              A small, CPU-only learning exercise: a logistic-regression
              baseline trained on simulated signals, with an &ldquo;uncertain&rdquo;
              answer for windows unlike its training data. Not a validated RF
              receiver, not an electronic-warfare system.
            </p>
            <p className="mt-3 text-sm text-ink-faint">
              Uploaded recordings are analyzed in memory and never stored.
              Upload only recordings you are allowed to share with the server.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
