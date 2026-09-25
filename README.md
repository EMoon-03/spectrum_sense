# SpectrumSense: interactive signal lab

This small, CPU-only project generates **simulated** complex I/Q samples for
noise, BPSK, and QPSK. It plots representative signals, extracts a few
explainable features, and trains a logistic-regression baseline. It is a
learning exercise, not a validated RF receiver or an EW system.

## Open the interactive app

Download `SpectrumSense_Interactive.zip` to your computer and extract it first.
In a terminal, change into the extracted `spectrum_sense` folder. Requires
Python 3.10 or newer. On Linux, these commands work in fish or Bash without
activating the virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local address Streamlit prints (usually `http://localhost:8501`) in
your browser. Press Ctrl+C in the terminal to stop the app. In VS Code you can
open this folder and select `.venv/bin/python` as the Python interpreter.

The app has three tabs:

- **Explore a signal:** choose noise, BPSK, or QPSK; adjust SNR, carrier offset,
  and random seed; inspect zoomable plots and the model scores. Download your
  example as a CSV.
- **Try your own recording:** upload a recording with 1,024 to 100,000 I/Q
  samples (see the formats below) and inspect any complete 1,024-sample
  window. The downloaded demo CSV works here as a first example.
- **How the model works:** inspect its grouped test accuracy and learn what the
  synthetic score does and does not establish.

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
raw SDR, NumPy, and SigMF readers, and feature extraction and app uploads for
extreme amplitudes.

## Run the command-line baseline

In the same folder:

```bash
.venv/bin/python spectrum_sense.py
```

If you prefer activating the environment in fish, use
`source .venv/bin/activate.fish`. On Windows PowerShell, use
`.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Look in `results/` for `example_signals.png`, `confusion_matrix.png`,
`metrics.txt`, and `capture_manifest.csv`. To change the dataset size:

```bash
.venv/bin/python spectrum_sense.py --captures-per-class 80 --windows-per-capture 4
```

## What to inspect first

1. Open `example_signals.png`. I and Q are the real and imaginary parts of the
   received complex samples. The FFT plot shows signal energy by frequency.
2. Read `capture_manifest.csv`: each row is one recording session. The `split`
   column assigns a complete session to training or testing.
3. Read `metrics.txt` and `confusion_matrix.png`. Find the least reliable class
   and the signal-to-noise ratio (SNR) range where predictions fail.
4. In `spectrum_sense.py`, adjust `SNR_DB_RANGE` or `FREQ_OFFSET_RANGE`, rerun,
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
fixed seed of 42. The `results/` folder in this starter archive contains one
example run so you can compare its figures with your own output.

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
After it is reproducible, add 16-QAM, channel fading, and unknown-signal
rejection as separate experiments.
