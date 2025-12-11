#!/usr/bin/env python3
"""
Phase 4: Modeling and Cross-Validation
Trains XGBoost model and compares with OLS and Random Forest baselines.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not installed. Using RandomForest as primary model.")

from config import (
    OUTPUT_DIR, MODELS_DIR, RANDOM_SEED, CV_FOLDS,
    XGBOOST_PARAMS, RF_PARAMS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


def load_splits():
    """Load train/val/test splits from Phase 3."""
    
    splits = {}
    for split in ['train', 'val', 'test']:
        X = pd.read_csv(OUTPUT_DIR / f"{split}_X.csv")
        y = pd.read_csv(OUTPUT_DIR / f"{split}_y.csv").iloc[:, 0]
        splits[f'X_{split}'] = X
        splits[f'y_{split}'] = y
    
    logger.info(f"Loaded splits: Train={len(splits['X_train'])}, Val={len(splits['X_val'])}, Test={len(splits['X_test'])}")
    return splits


def train_ols(X_train, y_train):
    """Train OLS baseline model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """Train XGBoost model with optional early stopping."""
    
    if not HAS_XGBOOST:
        logger.warning("XGBoost not available, returning None")
        return None
    
    model = XGBRegressor(**XGBOOST_PARAMS)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X, y, model_name="Model"):
    """Evaluate model performance."""
    
    y_pred = model.predict(X)
    
    metrics = {
        'model': model_name,
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'mape': np.mean(np.abs((y - y_pred) / y)) * 100
    }
    
    return metrics, y_pred


def cross_validate(model, X, y, cv=CV_FOLDS):
    """Perform k-fold cross-validation."""
    
    kfold = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    
    scores = {
        'r2': cross_val_score(model, X, y, cv=kfold, scoring='r2'),
        'neg_rmse': cross_val_score(model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
    }
    
    return {
        'r2_mean': scores['r2'].mean(),
        'r2_std': scores['r2'].std(),
        'rmse_mean': -scores['neg_rmse'].mean(),
        'rmse_std': scores['neg_rmse'].std()
    }


def compare_models(splits):
    """Train and compare all models."""
    
    X_train = splits['X_train']
    y_train = splits['y_train']
    X_val = splits['X_val']
    y_val = splits['y_val']
    X_test = splits['X_test']
    y_test = splits['y_test']
    
    results = []
    models = {}
    
    # 1. OLS Baseline
    logger.info("\n--- Training OLS ---")
    ols = train_ols(X_train, y_train)
    models['ols'] = ols
    
    train_metrics, _ = evaluate_model(ols, X_train, y_train, "OLS-Train")
    test_metrics, ols_pred = evaluate_model(ols, X_test, y_test, "OLS-Test")
    
    cv_scores = cross_validate(LinearRegression(), X_train, y_train)
    
    results.append({
        'model': 'OLS',
        'train_r2': train_metrics['r2'],
        'test_r2': test_metrics['r2'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'cv_r2_mean': cv_scores['r2_mean'],
        'cv_r2_std': cv_scores['r2_std']
    })
    logger.info(f"OLS: R²={test_metrics['r2']:.3f}, RMSE={test_metrics['rmse']:.5f}")
    
    # 2. Random Forest
    logger.info("\n--- Training Random Forest ---")
    rf = train_random_forest(X_train, y_train)
    models['random_forest'] = rf
    
    train_metrics, _ = evaluate_model(rf, X_train, y_train, "RF-Train")
    test_metrics, rf_pred = evaluate_model(rf, X_test, y_test, "RF-Test")
    
    cv_scores = cross_validate(RandomForestRegressor(**RF_PARAMS), X_train, y_train)
    
    results.append({
        'model': 'RandomForest',
        'train_r2': train_metrics['r2'],
        'test_r2': test_metrics['r2'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'cv_r2_mean': cv_scores['r2_mean'],
        'cv_r2_std': cv_scores['r2_std']
    })
    logger.info(f"RF: R²={test_metrics['r2']:.3f}, RMSE={test_metrics['rmse']:.5f}")
    
    # 3. XGBoost
    if HAS_XGBOOST:
        logger.info("\n--- Training XGBoost ---")
        xgb = train_xgboost(X_train, y_train, X_val, y_val)
        models['xgboost'] = xgb
        
        train_metrics, _ = evaluate_model(xgb, X_train, y_train, "XGB-Train")
        test_metrics, xgb_pred = evaluate_model(xgb, X_test, y_test, "XGB-Test")
        
        cv_scores = cross_validate(XGBRegressor(**XGBOOST_PARAMS), X_train, y_train)
        
        results.append({
            'model': 'XGBoost',
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'cv_r2_mean': cv_scores['r2_mean'],
            'cv_r2_std': cv_scores['r2_std']
        })
        logger.info(f"XGBoost: R²={test_metrics['r2']:.3f}, RMSE={test_metrics['rmse']:.5f}")
    
    return pd.DataFrame(results), models


def get_feature_importance(model, feature_names, model_type='xgboost'):
    """Extract feature importance from model."""
    
    if model_type == 'xgboost' and HAS_XGBOOST:
        importance = model.feature_importances_
    elif model_type == 'random_forest':
        importance = model.feature_importances_
    elif model_type == 'ols':
        importance = np.abs(model.coef_)
        importance = importance / importance.sum()
    else:
        return None
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df


def save_models(models, results):
    """Save trained models and results."""
    
    # Save models
    for name, model in models.items():
        if model is not None:
            model_path = MODELS_DIR / f"{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved: {model_path}")
    
    # Save results
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    logger.info(f"Saved: {OUTPUT_DIR / 'model_comparison.csv'}")


def main():
    """Execute Phase 4."""
    logger.info("=" * 60)
    logger.info("PHASE 4: MODELING AND CROSS-VALIDATION")
    logger.info("=" * 60)
    
    # Load data
    splits = load_splits()
    
    # Train and compare models
    results, models = compare_models(splits)
    
    # Get feature importance
    feature_names = splits['X_train'].columns.tolist()
    
    if 'xgboost' in models and models['xgboost'] is not None:
        fi = get_feature_importance(models['xgboost'], feature_names, 'xgboost')
        fi.to_csv(OUTPUT_DIR / "feature_importance_xgboost.csv", index=False)
        logger.info(f"\nTop 5 Features (XGBoost):\n{fi.head()}")
    
    if 'random_forest' in models:
        fi_rf = get_feature_importance(models['random_forest'], feature_names, 'random_forest')
        fi_rf.to_csv(OUTPUT_DIR / "feature_importance_rf.csv", index=False)
    
    # Save
    save_models(models, results)
    
    logger.info("\n✅ Phase 4 Complete")
    logger.info(f"\nModel Comparison:\n{results.to_string(index=False)}")
    
    # Select best model
    best_model = results.loc[results['test_r2'].idxmax(), 'model']
    logger.info(f"\n🏆 Best Model: {best_model}")
    
    return models, results


if __name__ == "__main__":
    main()
