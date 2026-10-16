"""Build the response pieces that Explore and run windows share: the decision
for one window of samples and the three plots with their text summaries.

All numbers come from the same functions the Streamlit lab plots, so the two
clients can never disagree about a value.
"""

from __future__ import annotations

import math

import numpy as np

from causes import possible_causes
from signal_config import CLASSES, FEATURES, NOT_CLASSIFIABLE
from spectrum_sense import (
    FeatureEvidence,
    SignalClassifier,
    classify_features,
    explain_features,
    extract_features,
)
from summaries import (
    constellation_summary,
    decision_summary,
    evidence_summary,
    scores_summary,
    spectrum_db,
    spectrum_summary,
    time_summary,
)

from . import schemas

# The same view settings as the lab.
TIME_VIEW_SAMPLES = 128
CONSTELLATION_STEP = 2


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def display_order(classes) -> list[str]:
    """CLASSES order (noise, BPSK, QPSK), then any others: the order the lab,
    the CSV, and the API all present scores in, whatever order the
    classifier learned them in."""
    classes = list(classes)
    return [*(name for name in CLASSES if name in classes), *(name for name in classes if name not in CLASSES)]


def reorder(classes, row) -> list[float]:
    """A row of per-class values in display_order(classes)."""
    classes = list(classes)
    return [row[classes.index(name)] for name in display_order(classes)]


def decision_for(model: SignalClassifier, samples: np.ndarray) -> schemas.Decision:
    """Classify one window; an all-zero window is NOT_CLASSIFIABLE, not an error."""
    classes = display_order(model.classes)
    try:
        features = extract_features(samples)
    except ValueError as exc:
        return schemas.Decision(
            label=NOT_CLASSIFIABLE,
            reason=str(exc),
            leading=None,
            top_score=None,
            distance=None,
            distance_threshold=model.distance_threshold,
            min_top_score=model.min_top_score,
            classes=classes,
            scores=[None] * len(classes),
            summary=f"This window is {NOT_CLASSIFIABLE}: {exc}",
            scores_summary="No scores: the window's features cannot be computed.",
            causes=causes_out(samples, None),
        )
    all_scores, (decision,) = classify_features(model, features)
    scores = [float(score) for score in reorder(model.classes, all_scores[0])]
    evidence = explain_features(model, features)
    return schemas.Decision(
        label=decision.label,
        reason=decision.reason,
        leading=decision.leading,
        top_score=_finite(decision.top_score),
        distance=_finite(decision.distance),
        distance_threshold=model.distance_threshold,
        min_top_score=model.min_top_score,
        classes=classes,
        scores=scores,
        summary=decision_summary(decision, model.distance_threshold, model.min_top_score),
        scores_summary=scores_summary(classes, scores),
        evidence=evidence_out(evidence, model.distance_threshold),
        causes=causes_out(samples, features),
    )


def causes_out(samples: np.ndarray, features: np.ndarray | None) -> list[schemas.CauseOut]:
    return [
        schemas.CauseOut(key=cause.key, title=cause.title, detail=cause.detail)
        for cause in possible_causes(samples, features)
    ]


def evidence_out(evidence: FeatureEvidence, distance_threshold: float) -> schemas.EvidenceOut:
    return schemas.EvidenceOut(
        nearest_class=evidence.nearest,
        distance=evidence.distance,
        leading=evidence.leading,
        runner_up=evidence.runner_up,
        features=[
            schemas.FeatureEvidenceOut(
                name=name,
                description=description,
                value=float(evidence.values[index]),
                expected=float(evidence.expected[index]),
                distance_share=float(evidence.distance_shares[index]),
                distance_if_expected=float(evidence.distance_if_expected[index]),
                score_push=float(evidence.score_push[index]),
            )
            for index, (name, description) in enumerate(FEATURES)
        ],
        summary=evidence_summary(evidence, distance_threshold),
    )


def signal_view(
    samples: np.ndarray,
    sample_rate: float | None = None,
    center_frequency: float | None = None,
) -> schemas.SignalView:
    samples = np.asarray(samples, dtype=np.complex128)
    frequencies, levels = spectrum_db(samples)
    return schemas.SignalView(
        samples=schemas.Samples(i=samples.real.tolist(), q=samples.imag.tolist()),
        time_view_samples=min(TIME_VIEW_SAMPLES, len(samples)),
        constellation_step=CONSTELLATION_STEP,
        spectrum=schemas.Spectrum(
            frequencies=np.round(frequencies, 6).tolist(), levels_db=np.round(levels, 3).tolist()
        ),
        summaries=schemas.ViewSummaries(
            time=time_summary(samples, TIME_VIEW_SAMPLES),
            constellation=constellation_summary(samples, CONSTELLATION_STEP),
            spectrum=spectrum_summary(samples, sample_rate, center_frequency),
        ),
    )
