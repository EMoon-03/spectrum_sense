/** Shown while a page waits for the API. api.ts gives up after 8 s, so on a
 *  sleeping API this hands over to the waiting notice rather than staying. */
export default function Loading() {
  return (
    <main className="mx-auto max-w-6xl px-6 py-16">
      <p role="status" className="text-sm text-ink-faint">
        Loading…
      </p>
    </main>
  );
}
