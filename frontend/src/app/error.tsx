"use client";

import Link from "next/link";
import Notice from "@/components/Notice";

/** A page failed to render. The message stays plain: the error's text and
 *  stack belong in the server log, which `digest` points to, not on the
 *  page. `retry` fetches the page from the server again. */
export default function ErrorPage({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <main className="mx-auto max-w-6xl px-6 py-16">
      <div className="max-w-xl">
        <Notice tone="bad">
          <p>
            Something went wrong while showing this page. Trying again often
            works; if it doesn&rsquo;t, start again from the overview.
          </p>
          {error.digest && (
            <p className="mono mt-2 text-xs text-ink-faint">Reference {error.digest}</p>
          )}
        </Notice>
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => retry()}
            className="bg-accent px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90"
          >
            Try again
          </button>
          <Link
            href="/"
            className="border border-rule px-4 py-2 text-sm text-ink-muted transition-colors hover:border-accent hover:text-accent"
          >
            Overview
          </Link>
        </div>
      </div>
    </main>
  );
}
