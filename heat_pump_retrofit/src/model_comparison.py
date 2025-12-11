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


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder], List[str], List[str]]:
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

    return X, y, encoders, available_categorical, available_numeric


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


class TabularEmbeddingRegressor:
    """Lightweight neural network with learned embeddings implemented in NumPy."""

    def __init__(
        self,
        categorical_cols: List[str],
        continuous_cols: List[str],
        embedding_sizes: Dict[str, Tuple[int, int]],
        hidden_dims: Tuple[int, ...] = (128, 64),
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 50,
        patience: int = 5,
        random_state: int = 42,
    ) -> None:
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.embedding_sizes = embedding_sizes
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        self.embeddings: Dict[str, np.ndarray] = {}
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.cont_means: np.ndarray | None = None
        self.cont_stds: np.ndarray | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None

        rng = np.random.default_rng(self.random_state)
        for col, (cardinality, emb_dim) in self.embedding_sizes.items():
            self.embeddings[col] = rng.normal(0, 0.05, size=(cardinality, emb_dim)).astype(np.float32)

    def _initialize_layers(self, input_dim: int) -> None:
        layer_dims = (input_dim, *self.hidden_dims, 1)
        rng = np.random.default_rng(self.random_state)
        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            weight = rng.normal(0, np.sqrt(2 / max(1, in_dim)), size=(in_dim, out_dim)).astype(np.float32)
            bias = np.zeros((1, out_dim), dtype=np.float32)
            self.weights.append(weight)
            self.biases.append(bias)

    def _prepare_inputs(self, X: pd.DataFrame) -> Tuple[List[np.ndarray], np.ndarray]:
        cat_arrays = [X[col].values.astype(int) for col in self.categorical_cols]
        cont_data = X[self.continuous_cols].values.astype(np.float32) if self.continuous_cols else np.empty((len(X), 0), dtype=np.float32)
        if self.continuous_cols:
            cont_data = (cont_data - self.cont_means) / self.cont_stds
        return cat_arrays, cont_data

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

    def fit_with_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        sample_weight: np.ndarray | None = None,
        val_sample_weight: np.ndarray | None = None,
    ) -> "TabularEmbeddingRegressor":
        if self.continuous_cols:
            train_cont = X_train[self.continuous_cols].values.astype(np.float32)
            self.cont_means = train_cont.mean(axis=0)
            self.cont_stds = train_cont.std(axis=0) + 1e-6
        else:
            self.cont_means = None
            self.cont_stds = None

        embedding_dim_total = int(sum(dim for _, dim in self.embedding_sizes.values()))
        input_dim = embedding_dim_total + len(self.continuous_cols)
        self._initialize_layers(input_dim)

        y_train_array = y_train.values.astype(np.float32).reshape(-1, 1)
        y_val_array = y_val.values.astype(np.float32).reshape(-1, 1)

        self.target_mean = float(y_train_array.mean())
        self.target_std = float(y_train_array.std() + 1e-6)
        y_train_array = (y_train_array - self.target_mean) / self.target_std
        y_val_array = (y_val_array - self.target_mean) / self.target_std

        best_val_loss = float("inf")
        best_state: Tuple[Dict[str, np.ndarray], List[np.ndarray], List[np.ndarray]] | None = None
        epochs_no_improve = 0

        n_samples = len(X_train)
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            epoch_losses: List[float] = []
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                cat_batches, cont_batch = self._prepare_inputs(X_train.iloc[batch_idx])
                embeds = [self.embeddings[col][cats] for col, cats in zip(self.categorical_cols, cat_batches)]
                inputs = np.concatenate(embeds + [cont_batch], axis=1) if embeds else cont_batch

                activations = [inputs]
                pre_activations = []
                for W, b in zip(self.weights[:-1], self.biases[:-1]):
                    z = activations[-1] @ W + b
                    pre_activations.append(z)
                    activations.append(self._relu(z))

                # Output layer
                z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
                pre_activations.append(z_out)
                activations.append(z_out)

                targets = y_train_array[batch_idx]
                if sample_weight is not None:
                    weights = sample_weight[batch_idx].reshape(-1, 1).astype(np.float32)
                    norm = np.sum(weights) + 1e-8
                    loss_grad = 2 * weights * (activations[-1] - targets) / norm
                else:
                    loss_grad = 2 * (activations[-1] - targets) / len(batch_idx)

                grads_W: List[np.ndarray] = []
                grads_b: List[np.ndarray] = []
                delta = loss_grad
                # Backprop output layer
                grads_W.append(activations[-2].T @ delta)
                grads_b.append(delta.sum(axis=0, keepdims=True))

                # Backprop hidden layers
                for layer in range(len(self.hidden_dims) - 1, -1, -1):
                    delta = (delta @ self.weights[layer + 1].T) * self._relu_grad(pre_activations[layer])
                    grads_W.append(activations[layer].T @ delta)
                    grads_b.append(delta.sum(axis=0, keepdims=True))

                grads_W = grads_W[::-1]
                grads_b = grads_b[::-1]

                # Gradient for embeddings and continuous features portion of input
                delta_input = delta @ self.weights[0].T
                offset = 0
                for col, cats in zip(self.categorical_cols, cat_batches):
                    emb_dim = self.embedding_sizes[col][1]
                    grad_slice = delta_input[:, offset : offset + emb_dim]
                    grad_matrix = np.zeros_like(self.embeddings[col])
                    np.add.at(grad_matrix, cats, grad_slice)
                    grad_matrix += self.weight_decay * self.embeddings[col]
                    self.embeddings[col] -= self.lr * grad_matrix / len(batch_idx)
                    offset += emb_dim

                # Update first layer inputs for continuous portion is handled via weight update
                for i in range(len(self.weights)):
                    grads_W[i] += self.weight_decay * self.weights[i]
                    self.weights[i] -= self.lr * grads_W[i]
                    self.biases[i] -= self.lr * grads_b[i]

                batch_loss = float(np.mean((activations[-1] - targets) ** 2))
                epoch_losses.append(batch_loss)

            # Validation
            val_predictions = self.predict(X_val, training_mode=True)
            if val_sample_weight is not None:
                val_loss = float(
                    np.sum(val_sample_weight * (val_predictions - y_val_array.squeeze()) ** 2)
                    / (np.sum(val_sample_weight) + 1e-8)
                )
            else:
                val_loss = float(np.mean((val_predictions - y_val_array.squeeze()) ** 2))

            logger.info(
                "Epoch %d - train loss: %.4f, val loss: %.4f",
                epoch + 1,
                float(np.mean(epoch_losses)),
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = (
                    {col: emb.copy() for col, emb in self.embeddings.items()},
                    [w.copy() for w in self.weights],
                    [b.copy() for b in self.biases],
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        epochs_no_improve,
                    )
                    break

        if best_state is not None:
            self.embeddings, self.weights, self.biases = best_state
        return self

    def predict(self, X: pd.DataFrame, training_mode: bool = False) -> np.ndarray:
        if self.weights is None or not self.weights:
            raise ValueError("Model has not been trained yet.")

        cat_arrays, cont_data = self._prepare_inputs(X)
        embeds = [self.embeddings[col][cats] for col, cats in zip(self.categorical_cols, cat_arrays)]
        inputs = np.concatenate(embeds + [cont_data], axis=1) if embeds else cont_data

        activation = inputs
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            activation = self._relu(activation @ W + b)
        outputs = activation @ self.weights[-1] + self.biases[-1]
        preds = outputs.squeeze()
        if not training_mode and self.target_mean is not None and self.target_std is not None:
            preds = preds * self.target_std + self.target_mean
        return preds if training_mode else preds.astype(float)

    def save(self, path: Path) -> None:
        import joblib

        payload = {
            "embeddings": self.embeddings,
            "weights": self.weights,
            "biases": self.biases,
            "cont_means": self.cont_means,
            "cont_stds": self.cont_stds,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "categorical_cols": self.categorical_cols,
            "continuous_cols": self.continuous_cols,
            "embedding_sizes": self.embedding_sizes,
        }
        joblib.dump(payload, path)


def build_models(
    categorical_cols: List[str],
    continuous_cols: List[str],
    X: pd.DataFrame,
) -> Dict[str, object]:
    """Define baseline models (including deep embedding) to benchmark against XGBoost."""

    embedding_sizes = {
        col: (int(X[col].nunique()), min(50, (int(X[col].nunique()) + 1) // 2))
        for col in categorical_cols
    }

    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=15,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.15,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=40,
            eval_metric="rmse",
        ),
        "DeepEmbeddingNN": TabularEmbeddingRegressor(
            categorical_cols=categorical_cols,
            continuous_cols=continuous_cols,
            embedding_sizes=embedding_sizes,
            hidden_dims=(192, 96, 48),
            lr=8e-4,
            weight_decay=2e-5,
            batch_size=256,
            epochs=80,
            patience=10,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
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
    X, y, _, categorical_cols, continuous_cols = prepare_features(df)
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

    models = build_models(categorical_cols, continuous_cols, X)
    results: List[Dict[str, float]] = []

    for name, model in models.items():
        logger.info("Training %s model...", name)

        fit_params = get_fit_params(name, w_train)
        if hasattr(model, "fit_with_validation"):
            model.fit_with_validation(
                X_train,
                y_train,
                X_val,
                y_val,
                sample_weight=w_train,
                val_sample_weight=w_val,
            )
        elif name == "XGBoost":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=w_train,
                sample_weight_eval_set=[w_val] if w_val is not None else None,
                verbose=False,
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
        try:
            if hasattr(model, "save"):
                model_path = MODELS_DIR / f"{name.lower()}_thermal_intensity.pt"
                model.save(model_path)
            else:
                import joblib

                model_path = MODELS_DIR / f"{name.lower()}_thermal_intensity.joblib"
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
