"""Minimal, dependency-free example of a leakage-resistant OCR router.

This is an educational example, not the CXT-Select production implementation.
It demonstrates three boundaries:

1. assign folds by source group, never by crop;
2. make threshold decisions from out-of-fold calibration predictions;
3. keep test labels outside fit(), calibrate_threshold(), and route().
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Candidate:
    source_id: str
    crop_id: str
    engine: str
    text: str
    intrinsic_score: float


@dataclass(frozen=True)
class LabelledCandidate:
    candidate: Candidate
    error: float


@dataclass(frozen=True)
class Routed:
    candidate: Candidate
    estimated_error: float
    accepted: bool


def stable_group_fold(source_id: str, folds: int, salt: str = "ocr-router-v1") -> int:
    """Map every crop from one source to the same deterministic fold."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    digest = sha256(f"{salt}:{source_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


class SelectiveRouter:
    """Rank OCR candidates using calibration-only engine error estimates."""

    def __init__(self) -> None:
        self._engine_error: dict[str, float] | None = None
        self._threshold: float | None = None

    def fit(self, rows: Iterable[LabelledCandidate]) -> "SelectiveRouter":
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            engine = row.candidate.engine
            totals[engine] = totals.get(engine, 0.0) + row.error
            counts[engine] = counts.get(engine, 0) + 1
        if not counts:
            raise ValueError("fit requires labelled calibration rows")
        self._engine_error = {
            engine: totals[engine] / counts[engine] for engine in counts
        }
        return self

    def estimated_error(self, candidate: Candidate) -> float:
        if self._engine_error is None:
            raise RuntimeError("fit must be called first")
        prior = self._engine_error.get(candidate.engine, 1.0)
        # Intrinsic scores must be reference-free and fixed before test access.
        confidence_error = 1.0 - min(max(candidate.intrinsic_score, 0.0), 1.0)
        return 0.6 * prior + 0.4 * confidence_error

    def choose(self, candidates: Sequence[Candidate]) -> Candidate:
        if not candidates:
            raise ValueError("at least one candidate is required")
        return min(
            candidates,
            key=lambda c: (self.estimated_error(c), c.engine, c.text),
        )

    def freeze_threshold(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._threshold = threshold

    def route(self, candidates: Sequence[Candidate]) -> Routed:
        if self._threshold is None:
            raise RuntimeError("freeze_threshold must be called first")
        selected = self.choose(candidates)
        risk = self.estimated_error(selected)
        return Routed(selected, risk, risk <= self._threshold)


def grouped(rows: Iterable[LabelledCandidate]) -> dict[tuple[str, str], list[LabelledCandidate]]:
    result: dict[tuple[str, str], list[LabelledCandidate]] = {}
    for row in rows:
        key = (row.candidate.source_id, row.candidate.crop_id)
        result.setdefault(key, []).append(row)
    return result


def out_of_fold_predictions(
    rows: Sequence[LabelledCandidate], folds: int = 5
) -> list[tuple[float, float]]:
    """Return (estimated_error, observed_error) without in-fold fitting."""
    predictions: list[tuple[float, float]] = []
    by_crop = grouped(rows)
    for fold in range(folds):
        train = [
            row
            for row in rows
            if stable_group_fold(row.candidate.source_id, folds) != fold
        ]
        held_out_sources = {
            row.candidate.source_id
            for row in rows
            if stable_group_fold(row.candidate.source_id, folds) == fold
        }
        if not held_out_sources:
            continue
        router = SelectiveRouter().fit(train)
        for (source_id, _crop_id), crop_rows in by_crop.items():
            if source_id not in held_out_sources:
                continue
            selected = router.choose([row.candidate for row in crop_rows])
            observed = next(
                row.error for row in crop_rows if row.candidate == selected
            )
            predictions.append((router.estimated_error(selected), observed))
    if not predictions:
        raise ValueError("no out-of-fold predictions were produced")
    return predictions


def calibrate_threshold(
    predictions: Sequence[tuple[float, float]],
    candidate_thresholds: Sequence[float],
    max_conditional_error: float,
) -> float:
    """Choose maximum coverage, then smallest threshold, using OOF rows only."""
    feasible: list[tuple[int, float]] = []
    for threshold in candidate_thresholds:
        accepted = [observed for risk, observed in predictions if risk <= threshold]
        if not accepted:
            continue
        conditional_error = sum(accepted) / len(accepted)
        if conditional_error <= max_conditional_error:
            feasible.append((len(accepted), threshold))
    if not feasible:
        raise ValueError("no threshold satisfies the calibration constraint")
    _coverage_count, threshold = min(feasible, key=lambda item: (-item[0], item[1]))
    return threshold


def fit_frozen_router(
    calibration_rows: Sequence[LabelledCandidate],
    candidate_thresholds: Sequence[float],
    max_conditional_error: float,
    folds: int = 5,
) -> SelectiveRouter:
    """Calibrate OOF, refit on all calibration rows, and freeze the threshold."""
    oof = out_of_fold_predictions(calibration_rows, folds=folds)
    threshold = calibrate_threshold(oof, candidate_thresholds, max_conditional_error)
    final_router = SelectiveRouter().fit(calibration_rows)
    final_router.freeze_threshold(threshold)
    return final_router


def route_unlabelled_test(
    router: SelectiveRouter,
    candidates_by_crop: Mapping[str, Sequence[Candidate]],
) -> dict[str, Routed]:
    """Apply a frozen router. No test-label parameter exists by design."""
    return {
        crop_id: router.route(candidates)
        for crop_id, candidates in candidates_by_crop.items()
    }


