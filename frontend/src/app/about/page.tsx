import Link from "next/link";

export default function AboutPage() {
  return (
    <main>
      <section className="border-b border-rule bg-surface">
        <div className="mx-auto max-w-6xl px-6 pb-8 pt-10">
          <h1 className="serif text-4xl tracking-tight text-ink">Method</h1>
          <p className="mt-2 max-w-[68ch] leading-relaxed text-ink-muted">
            What the synthetic score establishes, what it does not, and the design decision the whole app is built around.
          </p>
        </div>
      </section>

      <div className="mx-auto max-w-[72ch] space-y-10 px-6 py-10 text-[15px] leading-relaxed text-ink">
        <section>
          <h2 className="serif text-xl">The central design decision</h2>
          <p className="mt-3 text-ink-muted">
            A logistic regression always picks one of its classes, often with a score near 1, even for a signal it never saw: simulated 8PSK and 16-QAM at 10 dB were called QPSK, and an unmodulated tone was called BPSK. A low-score threshold alone does not help. So the app answers &ldquo;uncertain&rdquo; for either of two reasons, checked before any class is named: the window is unlike the training data (its Mahalanobis distance to the nearest class is beyond a limit set on held-back training captures), or the scores are close (the top score is below 0.60).
          </p>
          <p className="mt-3 text-ink-muted">
            The API returns the reason and both limits with every decision, and the interface shows the comparison, because a class the model named and a class the model declined to name are different kinds of claim, and a bare word does not convey it.
          </p>
        </section>

        <section>
          <h2 className="serif text-xl">The data</h2>
          <p className="mt-3 text-ink-muted">
            Every training window is simulated: repeated ideal symbols at 8 samples per symbol with a random timing offset, carrier phase, a carrier offset within ±0.008 cycles per sample, and white noise at an SNR between −2 and 14 dB. Total power is normalized, so the model cannot separate signal from noise by reading power alone. Windows from one capture share a signal stream and impairments; the split is by capture, before training.
          </p>
        </section>

        <section>
          <h2 className="serif text-xl">What the score does not establish</h2>
          <p className="mt-3 text-ink-muted">
            Test accuracy reflects this generator and its impairment ranges. A very high score is expected from idealized timing and repeated symbols; it is not evidence of performance on real recordings. The bundled real recording (QPSK packets through an Intelsat 37e transponder, root-raised-cosine pulses at 4 samples per symbol) shows why: most of its windows mix a packet and a gap, their power jumps by 6 dB partway through, no training window does that, and the model rightly answers &ldquo;uncertain&rdquo;. Windows entirely inside a packet are answered &ldquo;uncertain&rdquo; too: their fourth-power line, the clearest sign of QPSK, is about half as strong as the model&rsquo;s own QPSK would show, because the pulses are smoothly shaped. Treat any answer on real data as a hint, not a confirmation.
          </p>
        </section>

        <section>
          <h2 className="serif text-xl">Why this answer</h2>
          <p className="mt-3 text-ink-muted">
            Every decision carries its evidence. The squared distance to the nearest trained class is split exactly among the eight features: the offset from the class&rsquo;s mean is whitened with the symmetric square root of its inverse covariance, which keeps each component as close as possible to its own feature, and the squared components add up to the squared distance. A share says where the distance lies, not what caused it; a feature with an ordinary value can hold a large share when it disagrees with the others. So each feature also gets the value the class&rsquo;s fit expects given the other seven, and the distance the window would have if only that feature took it: &ldquo;changing the fourth-power line alone would bring the window inside the limit&rdquo; is a statement you can check. For the scores, each feature&rsquo;s contribution to the log-odds of the leading class over the runner-up is exact, because the classifier is linear in the standardized features.
          </p>
          <p className="mt-3 text-ink-muted">
            For an &ldquo;uncertain&rdquo; window, three direct measurements name properties no training window has: the power changing within the window (a burst and a gap), a bare carrier, and a steady component at 0 Hz. Each states the number it measured and the margin its limit was chosen with on simulated windows. None changes a decision or names a modulation, and when none fires the page says so rather than guess.
          </p>
        </section>

        <section>
          <h2 className="serif text-xl">Preparing a recording</h2>
          <p className="mt-3 text-ink-muted">
            A real recording rarely matches the training timing. Three optional steps run before analysis: shift the signal&rsquo;s frequency to 0, keep a band around it with a linear-phase low-pass filter, and resample to 8 samples per symbol. Resampling to a lower rate helps (at 20 samples per symbol every window was uncertain as recorded; resampled by 2/5, most were named correctly). Resampling to a higher rate does not: interpolation fills only the old band, half of the new band holds no noise, and the model never saw that, so the app says how much of the band is filled whenever it is less than 95%.
          </p>
        </section>

        <section>
          <h2 className="serif text-xl">Accessibility</h2>
          <p className="mt-3 text-ink-muted">
            Each chart carries a text summary with the numbers it shows. I and Q differ by line style as well as colour, timeline decisions differ by row and marker shape, the chart colours are Okabe-Ito, and the spectrogram uses Cividis with a labelled colour bar. Every control has help text and a visible focus ring.
          </p>
        </section>

        <section>
          <h2 className="serif text-xl">Architecture</h2>
          <p className="mt-3 text-ink-muted">
            A FastAPI backend holds the DSP, the model, and an in-memory cache of analyzed runs; this Next.js frontend is one client of it, and the original Streamlit lab is another. The TypeScript types are generated from the API&rsquo;s OpenAPI document, so the two cannot disagree about a field. Every result carries a fingerprint of the model that produced it: a hash of the training settings, the source of every function that generates, measures, trains, or decides, and the versions of Python and the numerical libraries, with a test that fails if a new function escapes it.
          </p>
          <p className="mt-3 text-ink-muted">
            The numbers on the <Link href="/model" className="text-accent underline underline-offset-4">Model</Link> page are regression-tested: the repository&rsquo;s test suite repeats the measurements and fails if they drift.
          </p>
        </section>

        <section>
          <h2 className="serif text-xl">Credit</h2>
          <p className="mt-3 text-ink-muted">
            The bundled recording is an unchanged excerpt of &ldquo;Test of gr4-packet-modem through an Intelsat 37e C-band transponder&rdquo; by Daniel Estévez, Zenodo record 13371136, used under CC BY 4.0. SpectrumSense is an independent learning project and is not affiliated with any agency or company.
          </p>
        </section>
      </div>
    </main>
  );
}
