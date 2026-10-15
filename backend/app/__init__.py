"""SpectrumSense API.

The same DSP and model code that the Streamlit lab (streamlit_app.py) and the
command-line baseline (spectrum_sense.py) use, behind a typed HTTP boundary so
that any client can simulate a signal, analyze a recording, and read what the
model measured about itself.

Endpoint groups:

    /health     liveness, plus whether the model is trained and its fingerprint
    /model      the demo model's measurements and limits (the "About" tab)
    /explore    simulate one noise, BPSK, or QPSK window and classify it
    /analyze    upload a recording (or use the bundled one), prepare it,
                classify every window, and compute the spectrogram
    /runs       follow-up reads of an analysis: one window's plots and
                decision, or the per-window CSV
    /example    the bundled real recording: credit, license, and what it holds

Design rules carried over from the Streamlit app, now enforced by the
schemas rather than by the page:

- Every decision travels with its reason and the two limits it was checked
  against (the distance to the nearest trained class and the minimum top
  score), so a client can render "uncertain" without re-deriving thresholds.
- Every decision also carries its evidence (spectrum_sense.explain_features):
  where the distance to the nearest class lies, feature by feature, the value
  each feature would need to fit the others, and what pushed the scores; and,
  for an "uncertain" window, the measured possible causes (causes.py).
  Neither changes a decision.
- Whether an Explore setting lies inside the model's training range is a
  field of the response, not a note a page may forget to show.
- Every result carries the model fingerprint (app.model.MODEL_VERSION). The
  model is retrained from a fixed seed at every process start, so a change
  to the features or the training settings silently changes every answer;
  the fingerprint is what makes two per-window CSVs comparable.
- Uploaded samples are held in memory only for the duration of a run cache
  entry and are never written to disk (app/uploads.py keeps the form parser
  from spooling them to a temporary file). Error messages never repeat text
  from an uploaded file.

Routers are thin: all logic lives in the top-level modules (spectrum_sense,
prepare, iq_formats, spectrogram, summaries, explore_url, real_example),
which the regression tests in tests/ cover without an HTTP layer.
"""
