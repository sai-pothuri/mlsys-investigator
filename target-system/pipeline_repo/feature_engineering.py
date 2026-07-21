"""Feature engineering pipeline for churn classifier."""
import numpy as np


FEATURE_NAMES = [
    "account_age_days",
    "monthly_spend",
    "num_transactions_30d",
    "avg_transaction_value",
    "days_since_last_login",
    "support_tickets_90d",
    "product_category",
    "region",
    "device_type",
    "login_failure_rate",
    "session_duration_min",
    "referral_source",   # NEW: 0=organic, 1=paid, 2=partner, 3=unknown
]
EXPECTED_FEATURE_COUNT = len(FEATURE_NAMES)

CLIP_BOUNDS = {
    "account_age_days":     (0, 3650),
    "monthly_spend":        (0, 10000),
    "num_transactions_30d": (0, 500),
    "days_since_last_login": (0, 365),
    "support_tickets_90d":  (0, 50),
    "session_duration_min": (0, 120),
}

VALID_CATEGORICALS = {
    "product_category": {0, 1, 2},
    "region":           {0, 1, 2, 3, 4},
    "device_type":      {0, 1, 2},
    "referral_source":  {0, 1, 2, 3},
}


def validate_features(record: dict) -> dict:
    for field in FEATURE_NAMES:
        if field not in record:
            # referral_source defaults to 0 if absent (backwards compat)
            if field == "referral_source":
                record[field] = 0
            else:
                raise ValidationError(f"Missing required field: {field}")
    for cat, valid in VALID_CATEGORICALS.items():
        if record.get(cat) not in valid:
            import warnings
            warnings.warn(f"Unexpected value for {cat}: {record[cat]}")
    return record


def normalize_feature(name: str, value: float) -> float:
    if name in CLIP_BOUNDS:
        lo, hi = CLIP_BOUNDS[name]
        return float(np.clip(value, lo, hi))
    return value


def build_feature_vector(record: dict) -> np.ndarray:
    validate_features(record)
    features = [normalize_feature(f, record[f]) for f in FEATURE_NAMES]
    assert len(features) == EXPECTED_FEATURE_COUNT
    return np.array(features, dtype=np.float32)
