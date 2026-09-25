# SpectrumSense: interactive signal lab

SpectrumSense simulates complex I/Q samples for noise, BPSK, and QPSK, extracts
a few explainable features, and trains a logistic-regression baseline. A
Streamlit app lets you explore simulated signals and try your own recordings.
It is a small, CPU-only learning exercise on synthetic data, not a validated RF
receiver or an EW system.

## Quick start

You need Git and Python 3.10 or newer. The repository is private for now, so
cloning it needs a GitHub account with access to it. On Linux or macOS, these
commands work in Bash, zsh, or fish without activating the virtual environment:

```bash
git clone https://github.com/EMoon-03/spectrum_sense.git
cd spectrum_sense
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local address Streamlit prints (usually `http://localhost:8501`) in
your browser. Press Ctrl+C in the terminal to stop the app. On Debian or
Ubuntu, install the `python3-venv` package if creating the environment fails.

On Windows, in PowerShell:

```powershell
git clone https://github.com/EMoon-03/spectrum_sense.git
cd spectrum_sense
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

If you prefer to activate the environment, use `source .venv/bin/activate`
(Bash or zsh) or `source .venv/bin/activate.fish` (fish). In VS Code, open the
folder and select the `.venv` Python as the interpreter.

## Run it online (Streamlit Community Cloud)

Streamlit Community Cloud can host the app from GitHub, so others can use it in
a browser without installing anything. It deploys what is on GitHub, not what
is on your computer, so push your commits to `main` first.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub
   account, and authorize Streamlit when GitHub asks.
2. Because the repository is private, also give Community Cloud access to your
   private repositories (Streamlit's guide "Connect your GitHub account" shows
   where), or make the repository public first. See the note below.
3. Click **Create app**. When asked whether you already have an app, answer
   that you do, since the app is already on GitHub.
4. Fill in **Repository** `EMoon-03/spectrum_sense`, **Branch** `main`, and
   **Main file path** `app.py`. You can also choose the app's address on
   `streamlit.app` under **App URL**.
5. Open **Advanced settings** and choose a Python version. Choose 3.12, one of
   the two versions the automated tests run on. The app needs no secrets.
   Click **Save**.
6. Click **Deploy!** Community Cloud then installs `requirements.txt` and
   starts the app; the first build can take a few minutes.
7. Share the app's address. Later pushes to `main` update the running app.
   Explore-tab links work there too: add, for example,
   `?signal=QPSK&snr=3&offset=0.012&seed=7` to the app's address.

**Private repositories.** When this was written (September 2026), Streamlit's
documentation said that an app deployed from a private repository starts out
private, visible only to you and people you invite, and that you can have only
one private app at a time. Check the current Community Cloud documentation
before relying on that, or make the repository public if the app should be open
to everyone. Button names can also change; if the pages look different, follow
the deployment guide at docs.streamlit.io.

A hosted app runs on someone else's server: files uploaded in the app are sent
to that server. Upload only recordings you are allowed to share.

## What the app does

The app has three tabs:

- **Explore a signal:** choose noise, BPSK, or QPSK; adjust SNR (-12 to 20 dB),
  carrier offset (-0.03 to 0.03 cycles per sample), and random seed; inspect
  zoomable plots and the model's answer and scores. A note appears when the
  offset, or the SNR of a BPSK or QPSK signal, is outside the model's training
  range (-0.008 to 0.008 cycles per sample, and -2 to 14 dB). Download your
  example as a CSV. The page address holds the four settings, for example
  `http://localhost:8501/?signal=QPSK&snr=3&offset=0.012&seed=7`, so you can
  bookmark or share the exact example. A link's values are checked when it
  opens: numbers outside a control's range or between its steps move to the
  nearest allowed value, anything else falls back to the default, and a short
  note lists what changed. Each change also adds a browser history entry;
  going back changes the address but not the controls, so reload the page to
  open an earlier address.
- **Try your own recording:** upload a recording with 1,024 to 100,000 I/Q
  samples (see the formats below). Every complete 1,024-sample window is
  classified: a timeline shows the answer for each window, a table counts the
  answers, and you can download the per-window results (window number, first
  sample, answer, reason, distance, and score per class) as a CSV. Windows that
  cannot be classified, such as all-zero windows, are marked "not classifiable",
  and the tab says how many trailing samples after the last complete window
  were ignored. A spectrogram (waterfall) then shows the power at each
  frequency over the whole recording, in hertz and seconds when the file
  states a sample rate. It uses Hann-windowed 256-sample segments that overlap
  by half, averages neighboring segments so that at most 400 time steps are
  drawn, and gives levels in dB relative to the strongest cell, with a floor
  of -80 dB. Below that, you can plot and score any single window. The tab
  offers a fixed example CSV to download and upload as a first try; the CSV
  from the first tab works too.
- **How the model works:** inspect its grouped test accuracy, the limits behind
  the "uncertain" answer and how often it fires, and learn what the synthetic
  score does and does not establish.

### Accessibility

Each chart has a short text summary directly below it with the numbers it
shows, such as the model scores, the RMS of I and Q, the strongest frequency
bin, the runs of answers in the timeline, and how much the spectrogram's level
changes over time. Score bars print their values, the Q line is dashed so I and
Q never differ by color alone, timeline answers differ by row and marker shape,
the colors (Okabe-Ito blue and vermillion, plus gray) stay distinct with
color-vision deficiency, and every control has help text. The spectrogram uses
the Cividis color scale, which grows steadily lighter with level, has a labeled
color bar, and shows each cell's level on hover.

### The "uncertain" answer

The logistic regression always picks noise, BPSK, or QPSK, even for signals it
never saw, and often with a score near 1: in our measurements, simulated 8PSK
and 16-QAM at 10 dB SNR were called QPSK and an unmodulated tone was called
BPSK. A low-score threshold alone therefore does not help. The app answers
"uncertain" for either of two reasons:

1. **Unlike the training data.** The window's standardized features are too far
   from every trained class. Each class has a mean and a slightly shrunk
   covariance, and the distance is the Mahalanobis distance to the nearest
   class. A quarter of the training captures are held out of fitting, and the
   limit is the 97.5th percentile of their distances.
2. **Scores are close.** The top score is below 0.60. This weaker check
   catches some windows between trained classes.

The held-out test captures and the probe signals (8PSK, 16-QAM, and a tone,
from `generate_probe_capture`) were never used to choose these settings; the
app reports how often each is answered "uncertain". Expect about 5% of known
test windows to be answered uncertain. In our measurements the tone was flagged
at every SNR from -8 to 20 dB, 16-QAM mostly at 5 dB and above, and 8PSK only
from about 10 dB; at 0 to 5 dB almost no 8PSK windows were flagged. Signals far
below the training SNR range (for example -12 dB) are still mostly called
noise, and known signals above it are often flagged. This is a simple
heuristic tested only on synthetic probes, not validated unknown-signal
detection.

### Supported upload formats

| File | Contents |
| --- | --- |
| `.csv` | UTF-8 text with the header `I,Q` and one sample per row (at most 6 MB) |
| `.cf32`, `.fc32`, `.cfile` | Raw complex float32, little-endian (GNU Radio file sink) |
| `.cs16`, `.sc16` | Raw interleaved int16 I,Q, little-endian, scaled by 1/32768 |
| `.cs8` | Raw interleaved int8 I,Q (HackRF), scaled by 1/128 |
| `.cu8` | Raw interleaved uint8 I,Q (`rtl_sdr`), mapped by (x - 127.5) / 127.5 |
| `.npy` | A 1-D complex array, or a real array with two columns (I, Q) |
| `.sigmf` | An uncompressed SigMF archive holding one recording |
| `.sigmf-meta` + `.sigmf-data` | The two files of one SigMF recording, uploaded together |

SigMF recordings must be single-channel, with `core:datatype` `cf32_le`,
`ci16_le`, `ci8`, or `cu8`. The app shows the SigMF sample rate and center
frequency when the metadata includes them. Raw files carry no metadata. Either
way, the demo model assumes its own synthetic timing of 8 samples per symbol,
so a file's sample rate does not change what the model expects. Archives are
read in memory and never extracted; links, special files, and unsafe paths are
rejected. NumPy files are loaded without pickle support.

Most SDR captures hold far more than 100,000 samples. Trim a raw file before
uploading it; for example, `head -c 400000 capture.cs16 > short.cs16` keeps the
first 100,000 samples of a 4-byte-per-sample `.cs16` file.

CSV exports preserve the precision of the float64 I/Q components. These files
are larger than rounded exports: 100,000 samples take about 4.2 MB for typical
app data and at most 5.1 MB, so the 6 MB import limit fits any full-length
export. Blank lines are skipped, and invalid rows or CSV syntax are reported
with their line number in the file. Feature extraction normalizes a copy of
each window to avoid amplitude overflow and leaves the imported samples
unchanged. All-zero windows can be plotted but cannot be classified.

## Run the regression tests

With the dependencies installed, run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover CSV validation and limits, lossless float64 interchange, the
raw SDR, NumPy, and SigMF readers, feature extraction and app uploads for
extreme amplitudes, the "uncertain" answer and its probe signals, whole-file
classification and its CSV, the spectrogram, the shareable Explore links, and
the chart summaries and other accessibility features. The app tests run it
headless with Streamlit's `AppTest`, so no browser is needed.

GitHub Actions runs the same tests on Python 3.10 and 3.12 for every push and
pull request; see `.github/workflows/tests.yml`.

## Run the command-line baseline

In the same folder:

```bash
.venv/bin/python spectrum_sense.py
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Look in `results/` for `example_signals.png`, `confusion_matrix.png`,
`metrics.txt`, and `capture_manifest.csv`. To change the dataset size:

```bash
.venv/bin/python spectrum_sense.py --captures-per-class 80 --windows-per-capture 4
```

Add `--output-dir my_results` to write somewhere else and keep the example run
in `results/` unchanged.

## What to inspect first

1. Open `example_signals.png`. I and Q are the real and imaginary parts of the
   received complex samples. The FFT plot shows signal energy by frequency.
2. Read `capture_manifest.csv`: each row is one recording session. The `split`
   column assigns a complete session to training or testing.
3. Read `metrics.txt` and `confusion_matrix.png`. Find the least reliable class
   and the signal-to-noise ratio (SNR) range where predictions fail.
4. In `signal_config.py`, adjust `SNR_DB_RANGE` or `FREQ_OFFSET_RANGE`, rerun,
   and record what changed. This is a small, defensible first experiment.

The generator fixes total average power across examples, so the model cannot
solve the signal-versus-noise problem by reading power alone. Features include
second- and fourth-order complex moments, amplitude variability, and
symbol-spaced phase differences. Samples from one capture share a signal stream
and impairments; the script splits by capture ID **before** training to avoid putting sibling
windows into both sets. Test accuracy only reflects this particular synthetic
generator and its impairment ranges. A very high score here is expected from
idealized symbol timing and repeated symbols; it is **not** evidence of
performance on real recordings. Report it as a synthetic baseline.

The default run uses 60 captures per class, four windows per capture, and a
fixed seed of 42. The `results/` folder in this repository contains one example
run so you can compare its figures with your own output.

## Project files

| File | Contents |
| --- | --- |
| `app.py` | The Streamlit app |
| `spectrum_sense.py` | Signal generator, features, training, the "uncertain" check, whole-file classification, and the command-line baseline |
| `signal_config.py` | Shared settings: classes, window size, training ranges, and the "uncertain" limits |
| `iq_csv.py` | CSV import and lossless export |
| `iq_formats.py` | Readers for raw SDR, NumPy, and SigMF recordings |
| `spectrogram.py` | The whole-recording spectrogram |
| `summaries.py` | The text summaries under each chart |
| `explore_url.py` | Reads, checks, and writes the Explore tab's settings in shareable links |
| `tests/` | The regression tests |
| `results/` | One example run of the command-line baseline |

## Next increments

- **CS 513 / CS 556:** Add a short notebook explaining the dataset, phase,
  FFT, SNR, features, and the sources of errors. Compare a second baseline.
- **CS 559:** Reserve a validation split by capture, tune on validation only,
  then assess the test set once. Add an unseen frequency-offset or SNR test.
- **CS 583:** Add a small PyTorch model operating on two channels (I and Q),
  using exactly the same train/validation/test capture IDs.
- **DSP:** Replace the simplified `np.repeat` transmitter with pulse shaping
  and a more realistic channel, then compare against an independent recording
  source. GNU Radio's digital modulation tools are a natural next step.
- **Portfolio:** Include a concise results table, error cases, reproducible
  commands, a CLI for one recording, and the limits of simulated evaluation.

This project deliberately starts with three classes and a modest baseline.
After it is reproducible, add 16-QAM, channel fading, and a better-tested
unknown-signal check (new probe types, fresh seeds, and real recordings) as
separate experiments.
