"""The real recording that comes with the app: where it is, whom to credit,
and what it holds.

examples/intelsat37e_qpsk_excerpt.sigmf is an unchanged excerpt of a SigMF
recording by Daniel Estévez, shared under CC BY 4.0. tools/make_real_example.py
rebuilds it, and examples/ATTRIBUTION.md says how.
"""

from __future__ import annotations

from pathlib import Path

from signal_config import SAMPLES_PER_SYMBOL, SAMPLES_PER_WINDOW


EXAMPLE_PATH = Path(__file__).resolve().parent / "examples" / "intelsat37e_qpsk_excerpt.sigmf"
# Keep the repository small: the bundled file must stay below this size.
MAX_EXAMPLE_BYTES = 300_000

# The excerpt, counting samples from 0 in the original recording.
FIRST_SAMPLE = 204_000
SAMPLE_COUNT = 30 * SAMPLES_PER_WINDOW
LAST_SAMPLE = FIRST_SAMPLE + SAMPLE_COUNT - 1
ORIGINAL_SAMPLES = 596_633
SAMPLE_RATE = 500

TITLE = "Test of gr4-packet-modem through an Intelsat 37e C-band transponder"
AUTHOR = "Daniel Estévez"
RECORD = "13371136"
DOI = f"10.5281/zenodo.{RECORD}"
DOI_URL = f"https://doi.org/{DOI}"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
BLOG_URL = "https://destevez.net/2024/09/testing-gr4-packet-modem-through-a-geo-transponder/"

CREDIT = (
    f'**Credit:** "{TITLE}" by {AUTHOR}, Zenodo, [doi:{DOI}]({DOI_URL}), licensed under '
    f"[{LICENSE}]({LICENSE_URL}). This excerpt keeps samples {FIRST_SAMPLE:,} to "
    f"{LAST_SAMPLE:,} of the original recording unchanged; SpectrumSense added metadata "
    "that marks the packets."
)

# The packet numbers and the model's answers were measured on this excerpt, and
# tests/test_real_example.py checks them. The symbol rate, pulse shape, and
# syncword come from the author's blog post and the gr4-packet-modem source code.
ABOUT = (
    "**What it is.** Packets from an experimental modem, gr4-packet-modem, were sent "
    "through a C-band transponder on the Intelsat 37e satellite and recorded with a "
    "19 m antenna at a teleport in Usingen, Germany. The author describes the test in "
    f"[a blog post]({BLOG_URL}).\n\n"
    "- **Packets and gaps:** each packet lasts about 2,200 samples (4.4 seconds), and a "
    "new one starts about every 3,090 samples (6.2 seconds), so gaps of about 900 "
    "samples separate them. Packets are about 6 dB stronger than the gaps, which hold "
    "only noise. This excerpt starts and ends in a gap and holds ten complete "
    "packets.\n"
    "- **A BPSK start:** each packet opens with a 64-symbol BPSK syncword (256 samples) "
    "that helps the receiver find the packet; the rest of the packet is QPSK.\n"
    f"- **Different timing:** 125 symbols per second at {SAMPLE_RATE} samples per "
    "second gives 4 samples per symbol, with smoothly shaped (root-raised-cosine) "
    f"pulses. The model was trained on {SAMPLES_PER_SYMBOL} samples per symbol with "
    "rectangular pulses.\n"
    "- **The gaps:** the noise between packets is spread fairly evenly across the "
    "recorded band, much like the model's training noise. But every gap is shorter "
    f"than a {SAMPLES_PER_WINDOW:,}-sample window, so no window here holds gap alone.\n"
    '- **Why so many windows are "uncertain":** most windows hold part of a gap and '
    "part of a packet, so their power jumps by about 6 dB partway through. No training "
    'window does that, so the model answers "uncertain" because the window is unlike '
    "the training data. Windows that lie entirely inside a packet are mostly called "
    "QPSK, but they come close to the distance limit, so some are answered uncertain "
    "too. The model has seen only simulated signals: treat a QPSK answer as a hint, "
    "not a confirmation.\n"
    '- **Preparing it does not help:** under "Prepare the recording (optional)", a '
    f"symbol rate of 125 resamples the excerpt to {SAMPLES_PER_SYMBOL} samples per symbol "
    'and 60 windows, and every one of them is answered "uncertain" because it is unlike '
    "the training data. Resampling to a higher rate leaves half of the band without "
    "noise, which the model never saw."
)
