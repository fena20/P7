"""
Model Comparison for Thermal Intensity Prediction
=================================================

This module evaluates multiple regression models (including the existing
XGBoost baseline) on the prepared thermal intensity dataset and compares
performance across train, validation, and test splits.
"""

from pathlib import Path
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned dataset produced by the data prep pipeline."""

    filepath = OUTPUT_DIR / "03_gas_heated_clean.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            "Processed data not found. Please run 01_data_prep.py before model comparison."
        )
    return pd.read_csv(filepath)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """Prepare encoded feature matrix and target vector for modeling."""

    logger.info("Preparing features for model comparison...")

    numeric_features = [
        "HDD65",
        "A_heated",
        "building_age",
        "heating_equip_age",
        "log_sqft",
    ]

    categorical_features = [
        "TYPEHUQ",
        "YEARMADERANGE",
        "DRAFTY",
        "ADQINSUL",
        "TYPEGLASS",
        "EQUIPM",
        "REGIONC",
        "DIVISION",
        "envelope_class",
        "climate_zone",
    ]

    available_numeric = [f for f in numeric_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]

    logger.info("Available numeric features: %s", available_numeric)
    logger.info("Available categorical features: %s", available_categorical)

    X = df[available_numeric + available_categorical].copy()
    y = df["Thermal_Intensity_I"].copy()

    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    encoders: Dict[str, LabelEncoder] = {}
    for col in available_categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna("missing").astype(str))
        encoders[col] = le

    for col in available_numeric:
        X[col] = X[col].fillna(X[col].median())

    logger.info("Final feature matrix shape: %s", X.shape)

    return X, y, encoders


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify_col: str = "REGIONC",
):
    """Split the dataset into train, validation, and test partitions."""

    logger.info("Splitting data into train, validation, and test sets...")

    valid_idx = y.notna()
    strat = df.loc[valid_idx.index, stratify_col].fillna(-1) if stratify_col in df.columns else None
    weights = df.loc[valid_idx.index, "NWEIGHT"].values if "NWEIGHT" in df.columns else None

    X_trainval, X_test, y_trainval, y_test, strat_trainval, _ = train_test_split(
        X,
        y,
        strat if strat is not None else y,
        test_size=test_size,
        random_state=42,
        stratify=strat if strat is not None else None,
    )

    if weights is not None:
        w_trainval, w_test = train_test_split(
            weights[valid_idx.values],
            test_size=test_size,
            random_state=42,
        )
    else:
        w_trainval, w_test = None, None

    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=adjusted_val_size,
        random_state=42,
        stratify=strat_trainval if strat is not None else None,
    )

    if w_trainval is not None:
        w_train, w_val = train_test_split(
            w_trainval,
            test_size=adjusted_val_size,
            random_state=42,
        )
    else:
        w_train, w_val = None, None

    logger.info("Train size: %d", len(X_train))
    logger.info("Validation size: %d", len(X_val))
    logger.info("Test size: %d", len(X_test))

    return (X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray = None,
    set_name: str = "Test",
) -> Dict[str, float]:
    """Compute standard regression metrics for the provided split."""

    y_pred = model.predict(X)
    metrics = {
        "set": set_name,
        "n_samples": len(y),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred),
        "mape": np.mean(np.abs((y - y_pred) / y)) * 100,
    }

    if sample_weight is not None:
        weighted_se = sample_weight * (y - y_pred) ** 2
        metrics["weighted_rmse"] = np.sqrt(np.sum(weighted_se) / np.sum(sample_weight))

    logger.info(
        "%s performance — RMSE: %.3f, MAE: %.3f, R²: %.3f, MAPE: %.2f%%",
        set_name,
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
        metrics["mape"],
    )
    return metrics


def build_models() -> Dict[str, object]:
    """Define baseline models to benchmark against XGBoost."""

    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        ),
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.0005, random_state=42, max_iter=5000)),
            ]
        ),
    }


def get_fit_params(model_name: str, sample_weight: np.ndarray) -> Dict[str, np.ndarray]:
    """Return the appropriate fit parameters for models supporting sample weights."""

    if sample_weight is None:
        return {}

    if model_name in {"XGBoost", "RandomForest", "GradientBoosting", "ExtraTrees"}:
        return {"sample_weight": sample_weight}

    if model_name in {"LinearRegression", "Ridge", "Lasso"}:
        return {"model__sample_weight": sample_weight}

    return {}


def train_and_evaluate_models():
    """Run the full comparison workflow and save results to disk."""

    df = load_processed_data()
    X, y, _ = prepare_features(df)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        w_train,
        w_val,
        w_test,
    ) = split_data(X, y, df)

    models = build_models()
    results: List[Dict[str, float]] = []

    for name, model in models.items():
        logger.info("Training %s model...", name)

        fit_params = get_fit_params(name, w_train)
        if name == "XGBoost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                **fit_params,
            )
        else:
            model.fit(X_train, y_train, **fit_params)

        train_metrics = evaluate_model(model, X_train, y_train, w_train, "Train")
        val_metrics = evaluate_model(model, X_val, y_val, w_val, "Validation")
        test_metrics = evaluate_model(model, X_test, y_test, w_test, "Test")

        for metrics in (train_metrics, val_metrics, test_metrics):
            metrics["model"] = name
            results.append(metrics)

        # Persist trained model for reference
        model_path = MODELS_DIR / f"{name.lower()}_thermal_intensity.joblib"
        try:
            import joblib

            joblib.dump(model, model_path)
            logger.info("Saved %s model to %s", name, model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save %s model: %s", name, exc)

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "model_comparison_metrics.csv", index=False)

    # Summarize test set performance
    test_summary = (
        results_df[results_df["set"] == "Test"][
            ["model", "rmse", "mae", "r2", "mape"]
        ]
        .sort_values(by="rmse")
        .reset_index(drop=True)
    )
    test_summary.to_csv(TABLES_DIR / "model_comparison_test_summary.csv", index=False)

    logger.info("Model comparison complete. Test performance:\n%s", test_summary.to_string(index=False))

    return results_df, test_summary


if __name__ == "__main__":
    train_and_evaluate_models()
