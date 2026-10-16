"""Shared signal settings without plotting or machine-learning dependencies."""

CLASSES = ("noise", "BPSK", "QPSK")
SAMPLES_PER_SYMBOL = 8
SAMPLES_PER_WINDOW = 1024
SNR_DB_RANGE = (-2.0, 14.0)
FREQ_OFFSET_RANGE = (-0.008, 0.008)  # cycles per sample, relative to baseband
# Signal types the model never sees in training; they only test "uncertain".
PROBE_KINDS = ("8PSK", "16-QAM", "tone")

# extract_features's columns, in order: a short name and what the value shows.
FEATURES = (
    ("squared-signal line", "Strength of the spectral line in the squared signal; strong for BPSK."),
    ("fourth-power line", "Strength of the spectral line in the signal to the fourth power; strong for BPSK and QPSK."),
    ("amplitude spread", "Standard deviation of the amplitude over its mean; low for a constant envelope."),
    ("amplitude peakiness", "Mean fourth power of the amplitude over the squared mean power; 1 for a constant envelope, 2 for noise."),
    ("lag-1 correlation", "How alike neighboring samples are; high when each symbol spans several samples."),
    ("symbol-spaced correlation", "How alike samples one symbol apart are; high for an unmodulated carrier."),
    ("two-phase symbol steps", "Whether the phase steps between symbols are 0 or 180 degrees, as in BPSK."),
    ("four-phase symbol steps", "Whether the phase steps between symbols are multiples of 90 degrees, as in BPSK and QPSK."),
)
FEATURE_NAMES = tuple(name for name, _ in FEATURES)

# The "uncertain" answer and its reasons.
UNCERTAIN = "uncertain"
REASON_UNFAMILIAR = "unlike the training data"
REASON_CLOSE_SCORES = "scores are close"
# Whole-file results for a window whose features cannot be computed (all zeros).
NOT_CLASSIFIABLE = "not classifiable"

# Settings of the "uncertain" heuristic, chosen with training captures only.
# Share of training captures held back to set the distance threshold.
CALIBRATION_FRACTION = 0.25
# Each class covariance is shrunk this much toward a scaled identity matrix.
COVARIANCE_SHRINKAGE = 0.003
# Percentile of calibration-window distances used as the distance threshold.
DISTANCE_PERCENTILE = 97.5
# Answer "uncertain" when the highest class score is below this value.
MIN_TOP_SCORE = 0.6
