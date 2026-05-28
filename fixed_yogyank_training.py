"""

This version focuses on:
- separation of model logic from policy logic
- leakage-safe feature selection
- temporal validation
- frozen preprocessing artifacts
- simple per-row reason codes for audit review
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
DATA_PATH = "farmer_scoring_sample_yogyank_round1_final.csv"
ARTIFACT_DIR = Path("artifacts")
VERSION = "yogyank-baseline-v2"
SPLIT_YEAR = 2024

NUMERIC_FEATURES = [
    "land_area_acres",
    "historical_repayment_score",
    "annual_income_inr",
    "liability_ratio_pct",
    "rainfall_deviation_pct",
    "ndvi_score",
]

CATEGORICAL_FEATURES = [
    "district",
    "crop_type",
    "pm_kisan_status",
    "irrigation_type",
    "land_ownership",
    "soil_type",
    "sales_channel",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target_entitlement_score"
TIME_COLUMN = "application_year"
REQUIRED_COLUMNS = FEATURES + [TARGET_COLUMN, TIME_COLUMN]

FEATURE_LABELS = {
    "land_area_acres": "land area",
    "historical_repayment_score": "historical repayment score",
    "annual_income_inr": "annual income",
    "liability_ratio_pct": "liability ratio",
    "rainfall_deviation_pct": "rainfall deviation",
    "ndvi_score": "NDVI score",
    "district": "district",
    "crop_type": "crop type",
    "pm_kisan_status": "PM-Kisan status",
    "irrigation_type": "irrigation type",
    "land_ownership": "land ownership",
    "soil_type": "soil type",
    "sales_channel": "sales channel",
}


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    metrics: dict
    target_bounds: tuple[float, float]
    test_sample_reasons: list[dict]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    df = df.copy()
    df[TIME_COLUMN] = df[TIME_COLUMN].astype(int)
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def temporal_split(df: pd.DataFrame, split_year: int = SPLIT_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[TIME_COLUMN] < split_year].copy()
    test_df = df[df[TIME_COLUMN] == split_year].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Temporal split failed. Need rows before {split_year} for training and rows in {split_year} for validation."
        )

    return train_df, test_df


def bounded_predict(pipeline: Pipeline, X: pd.DataFrame, target_bounds: tuple[float, float]) -> np.ndarray:
    raw_predictions = pipeline.predict(X)
    return np.clip(raw_predictions, target_bounds[0], target_bounds[1])


def evaluate_model(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> TrainingArtifacts:
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET_COLUMN]

    pipeline.fit(X_train, y_train)
    target_bounds = (float(y_train.min()), float(y_train.max()))
    predictions = bounded_predict(pipeline, X_test, target_bounds)

    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_years": sorted(train_df[TIME_COLUMN].astype(int).unique().tolist()),
        "test_years": sorted(test_df[TIME_COLUMN].astype(int).unique().tolist()),
        "target_bounds": {
            "min_train_target": target_bounds[0],
            "max_train_target": target_bounds[1],
        },
    }

    test_sample_reasons = get_reason_codes(
        pipeline=pipeline,
        row=X_test.iloc[[0]],
        top_n=3,
    )

    return TrainingArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        target_bounds=target_bounds,
        test_sample_reasons=test_sample_reasons,
    )


def _transformed_feature_names(preprocessor: ColumnTransformer) -> Iterable[str]:
    return preprocessor.get_feature_names_out()


def _row_to_dense_array(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().ravel()
    return np.asarray(matrix).ravel()


def _base_feature_name(transformed_name: str) -> str:
    prefix, remainder = transformed_name.split("__", 1)
    if prefix == "num":
        return remainder

    for feature in CATEGORICAL_FEATURES:
        feature_prefix = f"{feature}_"
        if remainder.startswith(feature_prefix):
            return feature
    return remainder


def _reason_text(feature: str, row: pd.DataFrame, contribution: float) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    direction = "increased" if contribution >= 0 else "reduced"
    value = row.iloc[0][feature]

    if feature in NUMERIC_FEATURES:
        if feature == "land_area_acres":
            return f"{label.title()} of {value:.2f} acres {direction} the score."
        if feature == "annual_income_inr":
            return f"{label.title()} of INR {value:,.0f} {direction} the score."
        if feature == "liability_ratio_pct":
            return f"{label.title()} of {value:.2f}% {direction} the score."
        if feature == "rainfall_deviation_pct":
            return f"{label.title()} of {value:.2f}% {direction} the score."
        return f"{label.title()} of {value:.3f} {direction} the score."

    return f"{label.title()} = {value} {direction} the score."


def get_reason_codes(pipeline: Pipeline, row: pd.DataFrame, top_n: int = 3) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_row = preprocessor.transform(row[FEATURES])
    dense_row = _row_to_dense_array(transformed_row)

    feature_names = list(_transformed_feature_names(preprocessor))
    coefficients = np.asarray(model.coef_).ravel()

    grouped_contributions: dict[str, float] = {}
    for transformed_name, feature_value, coefficient in zip(feature_names, dense_row, coefficients):
        base_feature = _base_feature_name(transformed_name)
        grouped_contributions[base_feature] = grouped_contributions.get(base_feature, 0.0) + (
            float(feature_value) * float(coefficient)
        )

    ranked_features = sorted(grouped_contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    reason_codes = []
    for rank, (feature, contribution) in enumerate(ranked_features, start=1):
        reason_codes.append(
            {
                "rank": rank,
                "feature": feature,
                "contribution": round(float(contribution), 4),
                "direction": "positive" if contribution >= 0 else "negative",
                "reason": _reason_text(feature, row, contribution),
            }
        )

    return reason_codes


def save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_artifacts(df: pd.DataFrame, artifacts: TrainingArtifacts) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)

    schema = {
        "required_columns": REQUIRED_COLUMNS,
        "feature_columns": FEATURES,
        "target_column": TARGET_COLUMN,
        "time_column": TIME_COLUMN,
        "input_dtypes": {column: str(df[column].dtype) for column in REQUIRED_COLUMNS},
    }

    metadata = {
        "model_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "split_year": SPLIT_YEAR,
        "model_type": "Ridge",
        "preprocessing": {
            "numeric_imputation": "median",
            "numeric_scaling": "standard",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "one_hot_handle_unknown_ignore",
        },
        "excluded_columns": [
            "farmer_id",
            "defaulted_in_next_12_months",
        ],
        "policy_note": "Business policy must remain outside the model and be applied in a separate versioned layer.",
    }

    manifest = {
        "artifacts": [
            "artifacts/yogyank_pipeline.joblib",
            "artifacts/feature_schema.json",
            "artifacts/model_metadata.json",
            "artifacts/training_metrics.json",
            "artifacts/sample_reason_codes.json",
            "artifacts/artifact_manifest.json",
        ]
    }

    joblib.dump(artifacts.pipeline, ARTIFACT_DIR / "yogyank_pipeline.joblib")
    save_json(ARTIFACT_DIR / "feature_schema.json", schema)
    save_json(ARTIFACT_DIR / "model_metadata.json", metadata)
    save_json(ARTIFACT_DIR / "training_metrics.json", artifacts.metrics)
    save_json(ARTIFACT_DIR / "sample_reason_codes.json", artifacts.test_sample_reasons)
    save_json(ARTIFACT_DIR / "artifact_manifest.json", manifest)


def train_and_save(data_path: str = DATA_PATH) -> TrainingArtifacts:
    df = load_data(data_path)
    train_df, test_df = temporal_split(df, split_year=SPLIT_YEAR)
    pipeline = build_pipeline()
    artifacts = evaluate_model(pipeline, train_df, test_df)
    save_artifacts(df, artifacts)
    return artifacts


def main() -> None:
    artifacts = train_and_save()
    print(f"Model version: {VERSION}")
    print(f"Validation split: train<{SPLIT_YEAR}, test={SPLIT_YEAR}")
    print(f"Validation R2: {artifacts.metrics['r2']:.4f}")
    print(f"Validation MAE: {artifacts.metrics['mae']:.2f}")
    print("Top 3 sample reason codes:")
    for item in artifacts.test_sample_reasons:
        print(f"  {item['rank']}. {item['reason']} (contribution={item['contribution']})")


if __name__ == "__main__":
    main()
