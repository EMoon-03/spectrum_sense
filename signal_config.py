"""Shared signal settings without plotting or machine-learning dependencies."""

CLASSES = ("noise", "BPSK", "QPSK")
SAMPLES_PER_SYMBOL = 8
SAMPLES_PER_WINDOW = 1024
SNR_DB_RANGE = (-2.0, 14.0)
FREQ_OFFSET_RANGE = (-0.008, 0.008)  # cycles per sample, relative to baseband
# Signal types the model never sees in training; they only test "uncertain".
PROBE_KINDS = ("8PSK", "16-QAM", "tone")

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
