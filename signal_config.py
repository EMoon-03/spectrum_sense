"""Shared signal settings without plotting or machine-learning dependencies."""

CLASSES = ("noise", "BPSK", "QPSK")
SAMPLES_PER_SYMBOL = 8
SAMPLES_PER_WINDOW = 1024
SNR_DB_RANGE = (-2.0, 14.0)
FREQ_OFFSET_RANGE = (-0.008, 0.008)  # cycles per sample, relative to baseband
