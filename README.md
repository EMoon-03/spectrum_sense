# SpectrumSense

A signal classifier that says "uncertain" when it should. SpectrumSense
simulates complex I/Q samples for noise, BPSK, and QPSK, extracts a few
explainable features, trains a logistic-regression baseline, and serves it
behind a typed API with a Next.js front end: simulate a window, analyze your
own SDR recording, and read what the model measured about itself.

It is a small, CPU-only learning exercise on synthetic data, not a validated
RF receiver or an EW system.

**3 classes · 1,024-sample windows · 376 tests · one real recording (CC BY 4.0)**

---

## Contents

- [The problem](#the-problem)
- [The central design decision](#the-central-design-decision)
- [What it looks like in practice](#what-it-looks-like-in-practice)
- [Architecture](#architecture)
- [The API](#the-api)
- [Testing](#testing)
- [What I would do differently](#what-i-would-do-differently)
- [Running it](#running-it)
- [Reference](#reference): upload formats, preparing a recording, real radio data
- [Not built](#not-built)

---

## The problem

A logistic regression always picks one of its classes. Hand it a signal it
never saw and it still answers, often with a score near 1: in our
measurements, simulated 8PSK and 16-QAM at 10 dB were called QPSK and an
unmodulated tone was called BPSK. A low-score threshold alone does not help.

Every real recording is a signal the model never saw: different pulse shape,
different timing, bursts mixed with gaps. A tool that names a class for each
of those is confidently wrong most of the time.

## The central design decision

**A decision never travels without its reasons and the limits it was checked
against.**

Before any class is named, two checks run, and either makes the answer
"uncertain":

```
window features
   │
   ▼
standardize → Mahalanobis distance to the nearest trained class
   │
   ├── distance > limit ──────────────→ UNCERTAIN, "unlike the training data"
   │
   ▼
logistic regression → scores
   │
   ├── top score < 0.60 ─────────────→ UNCERTAIN, "scores are close"
   │
   └── otherwise ─────────────────────→ the top class
```

The limit is the 97.5th percentile of distances in a quarter of the training
captures held back from fitting. The held-out test captures and the probe
signals (8PSK, 16-QAM, a tone) were never used to choose it.

The API returns the label, the reason, the distance and its limit, the top
score and its minimum, and every class score, as separate fields, and the
interface shows the comparison. A score bar alone cannot convey the
difference between "the model named QPSK" and "the model declined"; a badge
and two numbers can.

**The cost.** The checks are a heuristic tested only on synthetic probes. A
tone is flagged nearly always, 16-QAM usually, and 8PSK only above about 10 dB
because at 8 samples per symbol with rectangular pulses its features sit close
to QPSK's. The Model page and `backend/tests/test_uncertain.py` report those
rates rather than hiding them.

## What it looks like in practice

Real responses from the API, trimmed to the relevant fields.

**A named class, with the limits it cleared.** `GET /explore?signal=QPSK&snr=3&offset=0.012&seed=7`

```json
{
  "settings": {"signal": "QPSK", "snr_db": 3.0, "offset": 0.012, "seed": 7},
  "in_training_range": {"snr": true, "offset": false},
  "model_version": "4f21335020a6029b",
  "decision": {
    "label": "QPSK",
    "reason": null,
    "distance": 3.17, "distance_threshold": 4.43,
    "top_score": 0.97, "min_top_score": 0.6,
    "classes": ["noise", "BPSK", "QPSK"],
    "scores": [0.00003, 0.029, 0.970],
    "summary": "The model names QPSK: the window is 3.17 from the nearest trained class (limit 4.43) and the top score is 0.97 (minimum 0.60)."
  }
}
```

`in_training_range.offset` is false: the carrier offset is outside the range
the model trained on, and the page flags it. The flag is a field, not a note
a page may forget to show.

**A refusal, with the limit it failed.** `GET /explore?signal=BPSK&snr=-2&seed=11`

```json
{
  "label": "uncertain",
  "reason": "unlike the training data",
  "leading": "BPSK",
  "top_score": 0.64, "min_top_score": 0.6,
  "distance": 4.52, "distance_threshold": 4.43,
  "summary": "Uncertain because the window is unlike the training data: it is 4.52 from the nearest trained class, beyond the limit of 4.43. The leading candidate is BPSK with a score of 0.64."
}
```

**A real recording, prepared, and the honest result.** `POST /analyze/example {"symbol_rate": 125}`

```json
{
  "run_id": "RL63SusOuGgllLr5",
  "model_version": "4f21335020a6029b",
  "preparation": {
    "steps": ["Resampled by 2/1 from 500 samples per second to 1 thousand samples per second, which gives 8 samples per symbol at 125 symbols per second."],
    "ratio": [2, 1], "samples_per_symbol": 8.0, "filled_share": 0.5,
    "band_note": "Only about 50% of the prepared band holds the recording's signal and noise; the higher sample rate left the rest empty. The model learned noise that fills the whole band, so expect more \"uncertain\" answers."
  },
  "counts": [{"label": "noise", "count": 0}, {"label": "BPSK", "count": 0},
             {"label": "QPSK", "count": 0}, {"label": "uncertain", "count": 60}]
}
```

All 60 windows are uncertain, every one of them 40+ distance units from the
nearest class. The model expects rectangular pulses in a full band; it was
given root-raised-cosine pulses in half a band. Declining is the right answer,
and the `band_note` says why before the user asks.

**Every result carries the model fingerprint.** The model is retrained from a
fixed seed at every process start, so an edit to the features or the training
settings silently changes every answer. `model_version` hashes the training
settings, `signal_config.py`, and the source of the feature, training, and
decision functions. It is in `/health`, every response, and the last column of
every per-window CSV, so two CSVs from different months are comparable or
visibly not.

## Architecture

**Backend** Python 3.10+, FastAPI, NumPy, SciPy, scikit-learn
**Frontend** Next.js 16, TypeScript, Tailwind CSS 4, Recharts
**Lab** Streamlit (the original interface, kept as a second client of the same core)
**Deployment** Vercel and Render

```
backend/
├── app/
│   ├── main.py              FastAPI app, CORS, lifespan training, /health
│   ├── config.py            settings from the environment
│   ├── model.py             the one demo model per process, MODEL_VERSION
│   ├── schemas.py           the API contract (Pydantic)
│   ├── views.py             a window's decision and three plots, with summaries
│   ├── analysis.py          analyze, re-prepare, serialize a run
│   ├── runs.py              in-memory run cache (nothing touches disk)
│   └── routers/             /model, /explore, /analyze + /runs, /example
├── spectrum_sense.py        generator, features, training, decide(), CLI baseline
├── signal_config.py         shared settings: classes, window, ranges, limits
├── prepare.py               shift, filter, resample; loudest windows
├── iq_formats.py            CSV, raw SDR, NumPy, WAV, SigMF readers
├── spectrogram.py           whole-recording spectrogram
├── summaries.py             the text under every chart
├── explore_url.py           shareable-link parsing with snapping and notes
├── real_example.py          the bundled recording's credit and description
├── streamlit_app.py         the Streamlit lab
├── tools/                   fetch and evaluate real recordings
├── examples/                the bundled recording and its attribution
└── tests/                   376 tests
frontend/
└── src/
    ├── app/                 /, /explore, /analyze, /model, /about
    ├── components/          DecisionBadge, ScoreBars, SignalPlots, Spectrogram,
    │                        WindowTimeline, PreparePanel, AnalyzePanel, ...
    └── lib/
        ├── api.ts           typed client
        ├── api.generated.ts types generated from the API's OpenAPI document
        └── format.ts        labels, tones, Cividis, number formats
```

**The routers are thin.** Every piece of logic is in the top-level modules,
which the regression tests cover without an HTTP layer. `routers/analyze.py`
handles the upload limit and maps `ValueError` to 422; `analysis.py` does the
work.

**Types are generated, not hand-written.** `npm run types` writes
`api.generated.ts` from `/openapi.json`, and `api.ts` re-exports from it. A
field renamed in `schemas.py` fails `npm run build` instead of appearing as a
runtime `undefined`. The generated file is committed so the frontend builds
without a running backend.

**The address is the state.** On the Explore page, `/explore?signal=QPSK&snr=3&offset=0.012&seed=7`
is a server-rendered page, so a link reproduces the example exactly and the
browser's Back and Forward buttons move between examples. The Streamlit lab
could only approximate this; its README had to explain that Back changed the
address but not the example.

**Runs live in memory and nowhere else.** An analysis returns every window's
decision and the spectrogram at once, then the client reads one window's plots
at a time, re-prepares with other settings, or downloads the CSV, all by run
id. The recording is held for 30 minutes (or until 32 newer runs push it out)
and never written to disk. A reload with `?run=` in the address reads the run
back; a forgotten run is a 404 whose message says to analyze again.

**Charts carry their own numbers.** Every chart has a text summary beneath it
from `summaries.py`: the RMS of I and Q, the strongest bin, the runs of
decisions in the timeline, whether the spectrogram's peak stays put, sweeps,
or hops. I is solid blue and Q is dashed vermillion; timeline rows differ by
position and marker shape; the spectrogram is Cividis on a canvas with a
labelled colour bar and hover values. Colours are Okabe-Ito throughout.

## The API

| Method and path | What it does |
| --- | --- |
| `GET /health` | `ok` once the model has trained, with its fingerprint; Render's health check |
| `GET /model` | training settings, the "uncertain" limits, held-out accuracy by SNR band, probe flag rates, and the Explore control ranges |
| `GET /explore?signal&snr&offset&seed` | one simulated window: decision, plots, summaries, `in_training_range`, and `notes` for any link value that was snapped or ignored |
| `GET /explore/csv?…` | the same window as `I,Q` text |
| `POST /analyze` | multipart: one recording (or a SigMF pair) plus optional preparation settings |
| `POST /analyze/example` | the bundled recording, same response |
| `GET /runs/{id}` | the analysis again |
| `POST /runs/{id}/prepare` | other settings for the same recording, same run id |
| `GET /runs/{id}/windows/{n}` | one window's level, decision, and plots |
| `GET /runs/{id}/results.csv` | per-window results with the preparation and `model_version` columns |
| `GET /example`, `GET /example/file` | the bundled recording's credit and the file itself |

Errors: 413 for an upload over the limit, 422 with the reader's own message
for a file or settings it rejects (never repeating text from the file), 404
for a forgotten run. The full document is at `/docs`.

## Testing

**376 tests**, all offline, run on Python 3.10 and 3.12 by GitHub Actions
alongside the frontend's lint and type-checked build.

- `test_api.py` covers the HTTP boundary: every route, every error code, the
  guarantees the schemas make (reasons and limits travel with decisions, NaN
  never crosses, the fingerprint is everywhere, file text is never echoed),
  CORS, cache eviction, and that the bundled recording gives the documented
  counts (8 QPSK and 22 uncertain; 60 uncertain after resampling).
- `test_uncertain.py` repeats the measurements the Model page shows: the
  false-reject rate, accuracy when answered, and the probe flag rates by SNR,
  with floors that fail the build if the model drifts.
- `test_app.py` and `test_startup_and_reruns.py` run the Streamlit lab
  headless with `AppTest`, including its shareable links and its caching.
- The readers, CSV round trips, preparation, spectrogram, summaries, link
  parsing, the real-data fetch tool (with every network request faked), and
  the bundled recording's byte-for-byte rebuild are each covered.

What the suite proves is that the pipeline runs and that the measured numbers
hold. It does not prove the model works on real signals; the bundled
recording is there to show that it mostly does not, and why.

## What I would do differently

**Put the API in first.** The Streamlit lab grew to 900 lines because it was
the UI, the cache, the state machine, and the only interface at once. Pulling
the API out took an afternoon because the DSP and model modules never
imported Streamlit, but the lab's `session_state` logic had to be re-derived
to design the run cache. A typed boundary from the start would have made the
Streamlit page a thin client too.

**Generate the client types from day one.** The lab had no client, so there
was nothing to drift. The moment there are two clients, hand-written types
are a liability.

**Decide early what a result's identity is.** The per-window CSV existed for
months with a `preparation` column and no model fingerprint. Any change to
`extract_features` would have invalidated every saved CSV invisibly.

## Running it

Requires Python 3.10+ and Node 20+. No API keys, no database.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
uvicorn app.main:app --reload
```

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

http://localhost:3000, with the API document at http://localhost:8000/docs.

After changing `backend/app/schemas.py`, with the backend running:
`npm run types` regenerates `src/lib/api.generated.ts`, then `npm run build`
checks every component against it.

The Streamlit lab still runs from `backend/`: `streamlit run streamlit_app.py`.
The command-line baseline is `python spectrum_sense.py`; see `results/` for
an example run.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS origins, comma-separated |
| `MODEL_SEED`, `CAPTURES_PER_CLASS`, `WINDOWS_PER_CAPTURE` | 42, 100, 4 | training settings; part of the fingerprint |
| `RUN_CACHE_SIZE`, `RUN_TTL_SECONDS` | 32, 1800 | how many analyses one process remembers, for how long |
| `MAX_UPLOAD_BYTES` | 8,000,000 | largest request body `/analyze` accepts |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | API base for the frontend; required in production |

### Deploying

`render.yaml` deploys the API on Render's free plan with `/health` as the
health check; set `ALLOWED_ORIGINS` to the frontend's exact origin. Deploy
`frontend/` on Vercel with `NEXT_PUBLIC_API_URL` set to the API's address.
The model trains in under a second at startup; the first cold start on the
free plan is dominated by importing scikit-learn and SciPy.

A hosted app runs on someone else's server: files uploaded to it are sent
there. Upload only recordings you are allowed to share.

---

## Reference

### Supported upload formats

| File | Contents |
| --- | --- |
| `.csv` | UTF-8 text with the header `I,Q` and one sample per row (at most 6 MB) |
| `.cf32`, `.fc32`, `.cfile` | Raw complex float32, little-endian (GNU Radio file sink) |
| `.cs16`, `.sc16` | Raw interleaved int16 I,Q, little-endian, scaled by 1/32768 |
| `.cs8` | Raw interleaved int8 I,Q (HackRF), scaled by 1/128 |
| `.cu8` | Raw interleaved uint8 I,Q (`rtl_sdr`), mapped by (x - 127.5) / 127.5 |
| `.npy` | A 1-D complex array, or a real array with two columns (I, Q) |
| `.wav` | Two channels, I (left) and Q (right): 8-bit unsigned or 16-, 24-, or 32-bit integer PCM, or 32- or 64-bit float, including the extensible WAV format. Integers are scaled to about -1 to 1 (8-bit values by (x - 128) / 128) |
| `.sigmf` | An uncompressed SigMF archive holding one recording |
| `.sigmf-meta` + `.sigmf-data` | The two files of one SigMF recording, uploaded together |

SigMF recordings must be single-channel, with `core:datatype` `cf32_le`,
`ci16_le`, `ci8`, or `cu8`. The app shows the SigMF sample rate and center
frequency when the metadata includes them. Raw files carry no metadata. Either
way, the demo model assumes its own synthetic timing of 8 samples per symbol,
so a file's sample rate does not change what the model expects unless you
resample the recording (see [Prepare a recording](#prepare-a-recording)).
Archives are read in memory and never extracted; links, special files, and
unsafe paths are rejected, and the `._` files that macOS `tar` adds are
ignored. NumPy files are loaded without pickle support. Error messages
describe a problem without repeating text from the file, such as names inside
an archive.

WAV is how KiwiSDR's `kiwirecorder.py` and PC programs such as SDR# and HDSDR
save I/Q recordings. The app joins every data chunk, so a file made with
`kiwirecorder.py --kiwi-wav`, which has a `kiwi` time stamp chunk before each
block of samples, works too; other chunks are skipped. The sample rate comes
from the header, which stores whole hertz, so it may be rounded: kiwirecorder
writes a KiwiSDR's rate of about 12 kHz as, for example, 11999. If the file has
an `auxi` chunk in the layout that SpectraVue used and other SDR programs, such
as SDRangel, also write, the app shows the center frequency it states;
otherwise the center frequency is unknown. The app refuses one-channel WAV
files (ordinary audio, not I/Q), compressed WAV files, big-endian RIFX files,
RF64 files (a variant for files over 4 GB), and files cut short partway through
a chunk. We checked the reader against a real KiwiSDR recording and WAV files
written by ffmpeg and Python's `wave` module, not against files from SDR#,
HDSDR, or SDR Console.

Most SDR captures hold far more than 100,000 samples. Trim a raw file before
uploading it; for example, `head -c 400000 capture.cs16 > short.cs16` keeps the
first 100,000 samples of a 4-byte-per-sample `.cs16` file. Cutting a WAV file
that way leaves a header that promises more data than the file holds, which the
app refuses. Trim it with ffmpeg instead:
`ffmpeg -i capture.wav -af atrim=end_sample=100000 -c:a pcm_f32le short.wav`
keeps the first 100,000 samples as 32-bit floats, without loss for 8-, 16-, and
24-bit files. ffmpeg does not copy the `auxi` chunk, so the center frequency is
lost, and it reads only the first data chunk of a `--kiwi-wav` file, so keep
KiwiSDR recordings short instead (see the KiwiSDR row below).

CSV exports preserve the precision of the float64 I/Q components. These files
are larger than rounded exports: 100,000 samples take about 4.2 MB for typical
app data and at most 5.1 MB, so the 6 MB import limit fits any full-length
export. Blank lines are skipped, and invalid rows or CSV syntax are reported
with their line number in the file. Feature extraction normalizes a copy of
each window to avoid amplitude overflow and leaves the imported samples
unchanged. All-zero windows can be plotted but cannot be classified.

### Prepare a recording

A real recording rarely matches the training data: its signal may sit away
from the center of the band, share the band with other signals, or have a
different number of samples per symbol. Under **Prepare the recording (optional)** on the Analyze page, three optional
steps from `backend/prepare.py` run in this order before any analysis:

1. **Carrier offset to remove** shifts the recording so that the frequency you
   enter, read from the spectrogram, moves to 0 Hz. A shift alone changes no
   answer, because the features already remove a steady frequency offset; it
   matters before the next two steps, which keep only the band around 0 Hz.
2. **Bandwidth to keep** runs a linear-phase low-pass filter (Hamming-windowed,
   at most 1,001 taps) that keeps that band around 0 Hz. Mirrored samples at
   each end keep the level steady up to the first and last sample.
3. **Symbol rate** resamples the recording to 8 samples per symbol when the
   file states its sample rate; otherwise, a **resampling factor** from 0.01 to
   8 does. SciPy's `resample_poly` changes the rate by the fraction up/down
   closest to the factor with both terms at most 100: 125 symbols per second at
   500 samples per second gives exactly 2/1, and RDS at 2,375 symbols per
   second from 250,000 samples per second gives 6/79, 0.07% off.

Frequencies are in hertz when the file states its sample rate and in cycles
per sample otherwise. The prepared recording must hold at least one
1,024-sample window; beyond 100,000 samples, only the first 100,000 are kept,
with a note.
The tab shows the new sample count, sample rate, and samples per symbol, and
describes every step in plain words. The description also fills the
`preparation` column of the per-window CSV (`none` when nothing was done),
whose windows and sample numbers then count the prepared recording. **Reset the
preparation** returns to the recording as loaded, and a new recording starts
from the defaults. **Jump to a loud window** lists the five windows with the
most power, relative to the median window, and moves the window slider to the
one you choose; in a recording of bursts, they are the most likely to hold a
signal.

What we measured, with rectangular BPSK and QPSK simulated at other timings,
SNRs from the training range, and carrier offsets whose change per symbol is
within the training range (100 captures of each type and timing;
`backend/tests/test_prepare.py` checks smaller versions of these runs):

- **Resampling to a lower rate helps.** At 20 samples per symbol, every window
  was answered "uncertain" as recorded. Resampled by 2/5 to 8 samples per
  symbol, 68% of BPSK and 81% of QPSK windows were named correctly, and the
  rest were uncertain.
- **Resampling to a higher rate does not.** At 4 samples per symbol, 57% of
  BPSK and 53% of QPSK windows were named correctly as recorded, and the rest
  were uncertain. Resampled by 2/1, every window was uncertain: interpolation
  fills only the old band, so half of the new band holds no noise, unlike the
  noise the model learned. A filter narrower than the band leaves part of it
  empty too, so the tab says how much of the prepared band is filled whenever
  that is less than 95%.

## Get real radio data

The demo model has only seen simulated signals, so real recordings are the
honest test of what it learned. You don't need a radio to start.

### The bundled real recording

The app comes with one real recording,
`backend/examples/intelsat37e_qpsk_excerpt.sigmf` (256 KB): samples 204,000 to
234,719 of Zenodo record 13371136, "Test of gr4-packet-modem through an
Intelsat 37e C-band transponder" by Daniel Estévez
([doi:10.5281/zenodo.13371136](https://doi.org/10.5281/zenodo.13371136)),
used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
30,720 samples (30 windows, about a minute at 500 samples per second) are
unchanged; the new SigMF metadata describes the excerpt and marks its ten
complete packets with annotations. The excerpt starts and ends in a gap between
packets. `backend/examples/ATTRIBUTION.md` gives the credit, what was changed, and how
the file was made; keep it with the file if you share the recording, which
stays under CC BY 4.0.

On the Analyze page, choose **Use the bundled satellite recording (QPSK, CC BY 4.0)**. `backend/tools/make_real_example.py` rebuilds the file
from Zenodo (it needs an internet connection and uses `backend/tools/fetch_real_iq.py`
below); the archive's dates are fixed, so the rebuild matches the bundled file
byte for byte.

### Download a real recording

`backend/tools/fetch_real_iq.py` downloads a SigMF recording from
[Zenodo](https://zenodo.org), checks it, and saves an excerpt the app can
read. It needs no account and uses only NumPy and the standard library. By
default it fetches Zenodo record 13371136, "Test of gr4-packet-modem through
an Intelsat 37e C-band transponder" by Daniel Estévez
([doi:10.5281/zenodo.13371136](https://doi.org/10.5281/zenodo.13371136),
CC BY 4.0): packets relayed by a geostationary satellite and recorded with a
USRP B200mini on the L-band IF of a teleport in Usingen (19 m antenna). The
file holds 596,633 samples (about 19.9 minutes at 500 samples per second,
4.8 MB) at a center frequency of 3,671,055,830 Hz.

```bash
python tools/fetch_real_iq.py                 # samples 0 to 99,999
python tools/fetch_real_iq.py --start 200000  # a part with longer packets
```

The tool asks the Zenodo API for the record, streams the data file (refusing
files over 50 MB unless you raise `--max-mb`), checks it against the MD5 that
Zenodo publishes and the SHA-512 in the SigMF metadata, and retries temporary
network failures. It keeps up to 100,000 samples from `--start` (`--samples`
chooses how many, at least 1,024) and writes four files to `real_iq/` (choose
another folder with `--out-dir`; `real_iq/` is in `.gitignore`):

| File | Contents |
| --- | --- |
| `NAME.sigmf` | A SigMF archive: the samples with their sample rate, center frequency, author, license, a description of the excerpt, and a new SHA-512. Upload this one; the app shows its sample rate and frequency. |
| `NAME.cf32` | The same samples as raw complex float32 |
| `NAME.csv` | The same samples as `I,Q` text, within the app's 6 MB CSV limit |
| `NAME.ATTRIBUTION.txt` | The source, its license, and what was changed. Keep it with the data. |

`NAME` is the recording's name and the sample range, for example
`gr4-packet-modem-intelsat37e-test_0-100000`. The tool also lists the loudest
and quietest 1,024-sample windows, which in this file are likely inside a
packet and between packets. `--record` fetches another Zenodo record with an
uncompressed `.sigmf-meta` and `.sigmf-data` pair (datatype `cf32_le`,
`ci16_le`, `ci8`, or `cu8`); `--recording` picks one if it holds several.

What the recording contains comes from the author's
[blog post](https://destevez.net/2024/09/testing-gr4-packet-modem-through-a-geo-transponder/)
and the [gr4-packet-modem](https://github.com/daniestevez/gr4-packet-modem)
README and source code, not from its metadata: uncoded QPSK at 125 symbols per
second (4 samples per symbol) with root-raised-cosine pulses, at an Es/N0 of
about 9.5 dB. Each packet opens with a 64-symbol BPSK syncword (256 samples),
then QPSK. We measured the rest with a smoothed-power detector: before sample
200,000, packets last about 1,300 samples and start about every 2,030 samples;
after it, they last about 2,200 samples and start about every 3,090. Packets
stand about 6 dB above the gaps between them, and about a third of the samples
are gaps. The noise in the gaps is spread fairly evenly across the recorded
band, much like the white noise the "noise" class was trained on.

### What to expect from the model

The model expects rectangular pulses at 8 samples per symbol; this recording
has root-raised-cosine pulses at 4, and a 1,024-sample window often mixes
packet and gap. An answer on it is an experiment, not a score. When we tried
it in September 2026, most windows came back "uncertain": 84 of the 97 windows
in samples 0 to 99,999, and 69 of 97 in samples 200,000 to 299,999. The rest
were called QPSK (11 and 28 windows, almost all in the loudest third) or noise
(2 windows, in the first excerpt); none was called BPSK. The quietest third of
the windows, mostly gaps, were nearly all uncertain. A QPSK answer is weak
evidence even so: before the "uncertain" answer existed, the loudest in-packet
window was called QPSK with a score of 0.975, but so was a recording of analog
FM.

The bundled excerpt shows why. Its gaps (about 900 samples) are shorter than a
window, so 20 of its 30 windows mix gap and packet: their power jumps by about
6 dB partway through, which no training window does, and all 20 are answered
"uncertain" as unlike the training data. Of the 10 windows that lie entirely
inside a packet, 8 are called QPSK and 2 are uncertain. Gap samples joined
into whole windows are nearly all called noise (146 of 150 such windows in our
test), but no window of the excerpt is gap alone.

Preparing the excerpt does not help. Resampled to 8 samples per symbol (a
symbol rate of 125), it holds 60 windows, and all of them are answered
"uncertain" as unlike the training data, well beyond the distance limit.
Moving the middle of its band, about 80 Hz below the center, to 0 Hz and
keeping 170 Hz around it before resampling leaves every window uncertain too,
and a shift alone changes no answer.

### Other sources of real I/Q

Checked in September 2026; services and their rules can change.

| Source | Access | What you get | License | Fit with the app |
| --- | --- | --- | --- | --- |
| Zenodo API | Free, no account: `GET https://zenodo.org/api/records/<id>`, files at `/api/records/<id>/files/<key>/content`. Range requests work, anonymous searches return at most 25 results, and rate limits apply. | Published datasets, some in SigMF | Set per record (13371136: CC BY 4.0) | Direct, with `backend/tools/fetch_real_iq.py` |
| [IQEngine](https://iqengine.org) | Its web app's backend (`https://iqengine.org/api/datasources/local/local/...`) needs no login but is undocumented, may change without notice, and refuses the default Python User-Agent (HTTP 403). | SigMF metadata and blocks of I/Q from community recordings | Most recordings state none | Good for browsing; without a license, keep excerpts to yourself |
| KiwiSDR | Public shortwave receivers over the internet, recorded with `kiwirecorder.py` from [kiwiclient](https://github.com/jks-prv/kiwiclient): `python3 kiwirecorder.py -s HOST -p 8073 -f <kHz> -m iq --tlimit=8 --user=YOURNAME` | A WAV file of 2-channel int16 I/Q at about 12 kHz (the header rounds the rate to a whole number, such as 11999); `--kiwi-wav` adds `kiwi` chunks between several data chunks | None: these are other people's transmissions | Direct: upload the WAV file, with or without `--kiwi-wav`. At about 12,000 samples per second, `--tlimit=8` stays under the 100,000-sample limit (a 10-second recording held 120,832 samples) |
| [SatNOGS Network](https://network.satnogs.org) | Free API, for example `https://network.satnogs.org/api/observations/?status=good&transmitter_mode=BPSK` | Satellite passes as demodulated audio (Ogg), not I/Q | CC BY-SA 4.0 | Indirect: a Hilbert transform of the audio gives pseudo-I/Q; in one test, a real 1200-baud BPSK packet made this way was classified as BPSK |
| [MIGOU-MOD](https://data.mendeley.com/datasets/fkwr8mzndr) (Mendeley Data, doi:10.17632/fkwr8mzndr.1) | Free; one 3.3 GB zip, described by `https://data.mendeley.com/public-api/datasets/fkwr8mzndr` | Real over-the-air recordings of RadioML-2016-style transmitters, labeled by modulation, BPSK included | CC BY 4.0 | Labeled real BPSK. The zip holds a Python pickle: read its float32 arrays with HTTP range requests, and never unpickle a downloaded file |
| Your own RTL-SDR | A dongle costs about $35 to $50. Arch Linux: `sudo pacman -S rtl-sdr`; Python: `pip install pyrtlsdr` (0.5.0) | `rtl_sdr` writes `.cu8`, which the app reads (trim it to 100,000 samples) | Yours to share, within the notes below | Full control over what you record; see the RDS recipe below |

A first real BPSK signal: FM broadcast stations can carry RDS, a BPSK
subcarrier at 57 kHz with 2,375 symbols per second. At 19,000 samples per
second each symbol spans 8 samples, the model's training timing. Record 10
seconds of a station (here 99.5 MHz) with `rtl_sdr -f 99.5e6 -s 250000 -g 30
-n 2500000 fm.cu8`, then run this with the project's Python (SciPy comes with
scikit-learn) and upload `rds.cf32`:

```python
from math import gcd

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly

fs = 250_000  # the rtl_sdr -s value
raw = (np.fromfile("fm.cu8", np.uint8).astype(np.float32) - 127.5) / 127.5
iq = (raw[0::2] + 1j * raw[1::2])[fs // 10 :]  # skip 0.1 s while the tuner settles
mpx = np.angle(iq[1:] * np.conj(iq[:-1]))  # FM demodulation
mpx = mpx * np.exp(-2j * np.pi * 57_000 * np.arange(len(mpx)) / fs)  # RDS to 0 Hz
mpx = lfilter(firwin(101, 7_500, fs=fs), 1.0, mpx)
step = gcd(19_000, fs)
rds = resample_poly(mpx, 19_000 // step, fs // step)[500:]  # 19,000 samples/s
rds[:100_000].astype(np.complex64).tofile("rds.cf32")
```

The Analyze page's preparation controls can shift, filter, and resample a recording,
but not FM-demodulate it, so this recipe needs Python. Matching timing is not
enough on its own: when we ran this recipe on one sample FM capture, every
window of the result was answered "uncertain". For QPSK, try the LRPT downlink
of a Meteor-M weather satellite (137.9 or 137.1 MHz, root-raised-cosine pulses
at 72,000 symbols per second) during a pass. ADS-B aircraft signals at 1090
MHz use pulse-position modulation, a good negative example: the right answer
is none of the three classes.

We don't recommend these for honest real-data tests: RadioML (synthetic, CC
BY-NC-SA, and called dated by its publisher DeepSig), ORACLE and POWDER (OFDM
signals labeled by device rather than modulation), and NIST mds2-2731 (its
downloads timed out when we tried).

### Credit, law, and etiquette

- **Credit.** CC BY 4.0 lets you share and adapt a recording if you credit
  the author, link the license, and say what you changed;
  `NAME.ATTRIBUTION.txt` records all three, and so does
  `backend/examples/ATTRIBUTION.md` for the bundled recording, whose credit the app
  also shows. CC BY-SA 4.0 (SatNOGS) also asks you to share adaptations under
  the same license. A recording that states no license, such as most on
  IQEngine or anything recorded on someone else's KiwiSDR, is not yours to
  republish.
- **Shared receivers and servers.** Each KiwiSDR has few listener slots, and
  its owner sets the rules: record short clips, identify yourself with
  `--user`, and don't hog a slot. Zenodo and the other APIs apply rate limits;
  the fetch tool makes at most three attempts per request and waits between
  them.
- **What you may receive.** In the United States, receiving broadcast,
  amateur (including APRS), and GMRS transmissions is lawful. Don't decode or
  publish the contents of other people's private communications, such as
  pager messages, DECT phone calls, or cellular or Wi-Fi payloads. Other
  countries differ, and some, such as the UK, are stricter. This is not legal
  advice.


## Not built

- No persistent runs: an analysis lives in one process's memory for 30
  minutes. A shareable `/runs/{id}` across restarts would need a store, and
  the recording would have to be kept, which the privacy promise above
  forbids without consent
- No frontend tests beyond lint and the type-checked build; the Streamlit lab
  keeps its `AppTest` coverage for that reason
- The model is the synthetic baseline: three classes, rectangular pulses, no
  fading. 16-QAM, pulse shaping, a validation split by capture, and a small
  PyTorch model on raw I/Q are the next experiments, each with the same
  train/validation/test capture ids
- Rule-based link parsing, so an unusual link value falls back to a default
  with a note rather than failing
