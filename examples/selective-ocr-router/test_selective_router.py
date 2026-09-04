import unittest

from selective_router import (
    Candidate,
    LabelledCandidate,
    SelectiveRouter,
    calibrate_threshold,
    fit_frozen_router,
    out_of_fold_predictions,
    route_unlabelled_test,
    stable_group_fold,
)


def row(source, crop, engine, score, error, text=None):
    candidate = Candidate(source, crop, engine, text or f"{engine}-{crop}", score)
    return LabelledCandidate(candidate, error)


class SelectiveRouterTests(unittest.TestCase):
    def test_all_crops_from_source_share_fold(self):
        fold = stable_group_fold("screen-17", 5)
        self.assertEqual(fold, stable_group_fold("screen-17", 5))

    def test_fit_requires_calibration_rows(self):
        with self.assertRaises(ValueError):
            SelectiveRouter().fit([])

    def test_oof_prediction_is_created_for_each_crop(self):
        rows = []
        for source in ["s1", "s2", "s3", "s4", "s5", "s6"]:
            rows.extend([
                row(source, f"{source}-c1", "engine-a", 0.9, 0.05),
                row(source, f"{source}-c1", "engine-b", 0.7, 0.30),
            ])
        predictions = out_of_fold_predictions(rows, folds=3)
        self.assertEqual(6, len(predictions))

    def test_threshold_uses_maximum_feasible_coverage(self):
        predictions = [(0.1, 0.0), (0.2, 0.1), (0.3, 0.9)]
        threshold = calibrate_threshold(predictions, [0.1, 0.2, 0.3], 0.1)
        self.assertEqual(0.2, threshold)

    def test_test_routing_accepts_no_labels(self):
        calibration = []
        for source in ["s1", "s2", "s3", "s4", "s5", "s6"]:
            calibration.extend([
                row(source, f"{source}-c1", "engine-a", 0.95, 0.02),
                row(source, f"{source}-c1", "engine-b", 0.60, 0.35),
            ])
        router = fit_frozen_router(calibration, [0.05, 0.10, 0.20], 0.05, folds=3)
        test = {
            "test-crop": [
                Candidate("test-source", "test-crop", "engine-a", "hello", 0.97),
                Candidate("test-source", "test-crop", "engine-b", "he11o", 0.55),
            ]
        }
        result = route_unlabelled_test(router, test)
        self.assertEqual("engine-a", result["test-crop"].candidate.engine)


if __name__ == "__main__":
    unittest.main()


