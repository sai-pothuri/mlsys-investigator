"""
Unit tests for model/train.py — ground truth label function.
"""

from __future__ import annotations
import numpy as np
import pytest

from model.train import (
    FEATURE_NAMES,
    GROUND_TRUTH_BIAS,
    GROUND_TRUTH_WEIGHTS,
    assign_labels,
    compute_label_score,
    generate_training_features,
)


class TestComputeLabelScore:
    def test_linearity(self):
        X = np.zeros((1, 12))
        score = compute_label_score(X)
        assert score[0] == pytest.approx(GROUND_TRUTH_BIAS)

    def test_manual_calculation(self):
        X = np.ones((1, 12))
        expected = GROUND_TRUTH_WEIGHTS.sum() + GROUND_TRUTH_BIAS
        assert compute_label_score(X)[0] == pytest.approx(expected)

    def test_output_shape(self):
        X = np.random.default_rng(0).standard_normal((100, 12))
        scores = compute_label_score(X)
        assert scores.shape == (100,)

    def test_high_login_failure_raises_score(self):
        base = np.zeros((1, 12))
        high_fail = base.copy()
        high_fail[0, 9] = 0.29  # login_failure_rate index=9, weight=1.5

        assert compute_label_score(high_fail)[0] > compute_label_score(base)[0]

    def test_enterprise_reduces_score(self):
        base = np.zeros((1, 12))
        enterprise = base.copy()
        enterprise[0, 6] = 2.0  # product_category index=6, weight=-0.3

        assert compute_label_score(enterprise)[0] < compute_label_score(base)[0]


class TestAssignLabels:
    def test_output_is_binary(self):
        rng = np.random.default_rng(0)
        X = generate_training_features(1000, rng)
        labels = assign_labels(X, rng=np.random.default_rng(1))
        assert set(labels).issubset({0, 1})

    def test_churn_rate_near_27_percent(self):
        rng = np.random.default_rng(42)
        X = generate_training_features(50_000, rng)
        labels = assign_labels(X, rng=np.random.default_rng(42))
        assert 0.22 < labels.mean() < 0.33, f"Churn rate {labels.mean():.3f} out of expected range"

    def test_deterministic_with_same_seed(self):
        X = np.random.default_rng(0).standard_normal((100, 12))
        labels_a = assign_labels(X, rng=np.random.default_rng(7))
        labels_b = assign_labels(X, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(labels_a, labels_b)

    def test_different_seeds_can_differ(self):
        X = np.random.default_rng(0).standard_normal((500, 12))
        labels_a = assign_labels(X, rng=np.random.default_rng(1))
        labels_b = assign_labels(X, rng=np.random.default_rng(999))
        assert not np.array_equal(labels_a, labels_b)

    def test_high_login_failure_increases_churn(self):
        rng = np.random.default_rng(0)
        n = 5000
        # Low login_failure_rate (index 9)
        X_low = generate_training_features(n, rng)
        X_low[:, 9] = 0.01

        X_high = X_low.copy()
        X_high[:, 9] = 0.29

        labels_low = assign_labels(X_low, rng=np.random.default_rng(42))
        labels_high = assign_labels(X_high, rng=np.random.default_rng(42))
        assert labels_high.mean() > labels_low.mean()

    def test_scores_consistent_with_weights(self):
        X = np.ones((10, 12))
        scores = compute_label_score(X)
        labels_no_noise = (scores > 0).astype(int)
        # With noise_std=0, labels should exactly match thresholded score
        labels = assign_labels(X, noise_std=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(labels, labels_no_noise)


class TestGenerateTrainingFeatures:
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        X = generate_training_features(1000, rng)
        assert X.shape == (1000, 12)

    def test_feature_count_matches_names(self):
        rng = np.random.default_rng(0)
        X = generate_training_features(100, rng)
        assert X.shape[1] == len(FEATURE_NAMES)

    def test_non_negative_clipped_features(self):
        rng = np.random.default_rng(0)
        X = generate_training_features(10000, rng)
        non_neg_indices = [0, 2, 4, 5, 10]  # age, transactions, days_login, tickets, session
        for idx in non_neg_indices:
            assert X[:, idx].min() >= 0.0, f"Feature {FEATURE_NAMES[idx]} has negative values"

    def test_categorical_features_in_valid_range(self):
        rng = np.random.default_rng(0)
        X = generate_training_features(5000, rng)
        # product_category: {0,1,2}, region: {0,1,2,3,4}, device_type: {0,1,2}, referral_source: {0,1,2,3}
        cat_specs = [(6, {0, 1, 2}), (7, {0, 1, 2, 3, 4}), (8, {0, 1, 2}), (11, {0, 1, 2, 3})]
        for idx, valid in cat_specs:
            vals = set(X[:, idx].astype(int).tolist())
            assert vals.issubset(valid), f"{FEATURE_NAMES[idx]} has invalid values: {vals - valid}"

    def test_login_failure_rate_in_range(self):
        rng = np.random.default_rng(0)
        X = generate_training_features(10000, rng)
        lfr = X[:, 9]  # login_failure_rate
        assert lfr.min() >= 0.0
        assert lfr.max() <= 0.3
