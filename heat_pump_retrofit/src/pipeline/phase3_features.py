#!/usr/bin/env python3
"""
Phase 3: Data Processing and Feature Engineering
Creates derived features and prepares train/val/test splits.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import (
    OUTPUT_DIR, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    OUTLIER_PERCENTILES, ENVELOPE_CLASSES, CLIMATE_ZONES, TARGET
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


def load_data():
    """Load processed data from Phase 2."""
    filepath = OUTPUT_DIR / "03_gas_heated_clean.csv"
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} records from {filepath}")
    return df


def remove_outliers(df, columns=None, percentiles=OUTLIER_PERCENTILES):
    """Remove outliers based on percentiles."""
    
    if columns is None:
        columns = ['HDD65', 'TOTCSQFT', 'thermal_intensity']
    
    columns = [c for c in columns if c in df.columns]
    
    n_before = len(df)
    
    for col in columns:
        lower = df[col].quantile(percentiles[0] / 100)
        upper = df[col].quantile(percentiles[1] / 100)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    n_after = len(df)
    logger.info(f"Outlier removal: {n_before:,} -> {n_after:,} ({n_before - n_after:,} removed)")
    
    return df


def create_derived_features(df):
    """Create engineered features."""
    
    logger.info("Creating derived features...")
    
    # Log transform of square footage
    if 'TOTCSQFT' in df.columns:
        df['log_sqft'] = np.log(df['TOTCSQFT'].clip(lower=100))
    
    # Building age (if not already present)
    if 'building_age' not in df.columns and 'YEARMADERANGE' in df.columns:
        df['building_age'] = 2024 - df['YEARMADERANGE']
    
    # Interaction features
    if 'HDD65' in df.columns:
        if 'TOTCSQFT' in df.columns:
            df['hdd_sqft'] = df['HDD65'] * df['TOTCSQFT'] / 1e6
        
        if 'building_age' in df.columns:
            df['age_hdd'] = df['building_age'] * df['HDD65'] / 1e4
        
        df['sqft_per_hdd'] = df.get('TOTCSQFT', df.get('log_sqft', 1)) / df['HDD65'].clip(lower=1)
        df['hdd_squared'] = df['HDD65'] ** 2 / 1e7
    
    # Envelope class (if not present)
    if 'envelope_class' not in df.columns and 'envelope_score' in df.columns:
        df['envelope_class'] = pd.cut(
            df['envelope_score'],
            bins=[0, 0.4, 0.7, 1.0],
            labels=['poor', 'medium', 'good']
        )
    
    # Climate zone
    if 'HDD65' in df.columns and 'climate_zone' not in df.columns:
        df['climate_zone'] = pd.cut(
            df['HDD65'],
            bins=[0, 4000, 5500, 12000],
            labels=['Mild', 'Moderate', 'Cold']
        )
    
    # Encode categorical variables
    categorical_cols = ['envelope_class', 'climate_zone', 'division_name']
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))
    
    logger.info(f"Created features. Total columns: {len(df.columns)}")
    
    return df


def prepare_feature_matrix(df):
    """Prepare feature matrix X and target y."""
    
    # Define feature columns
    feature_cols = [
        'HDD65', 'log_sqft', 'building_age', 'envelope_score',
        'hdd_sqft', 'age_hdd', 'sqft_per_hdd', 'hdd_squared',
        'envelope_class_encoded', 'climate_zone_encoded'
    ]
    
    # Filter to available columns
    available_features = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available_features)} features: {available_features}")
    
    # Additional numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    additional = ['TYPEHUQ', 'ADQINSUL', 'DRAFTY', 'REGIONC']
    for col in additional:
        if col in numeric_cols and col not in available_features:
            available_features.append(col)
    
    # Create X and y
    X = df[available_features].copy()
    
    if TARGET in df.columns:
        y = df[TARGET].copy()
    else:
        logger.warning(f"Target '{TARGET}' not found, using synthetic target")
        y = pd.Series(np.random.uniform(0.003, 0.012, len(df)))
    
    # Handle any remaining NaN
    X = X.fillna(X.median())
    
    return X, y, available_features


def split_data(X, y, stratify_col=None):
    """Split data into train/val/test sets."""
    
    logger.info(f"Splitting data: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED
    )
    
    # Second split: val vs test
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=RANDOM_SEED
    )
    
    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


def save_processed_data(df, splits, features):
    """Save processed data and splits."""
    
    # Save full processed data
    df.to_csv(OUTPUT_DIR / "04_processed_features.csv", index=False)
    
    # Save splits
    splits['X_train'].to_csv(OUTPUT_DIR / "train_X.csv", index=False)
    splits['y_train'].to_csv(OUTPUT_DIR / "train_y.csv", index=False)
    splits['X_val'].to_csv(OUTPUT_DIR / "val_X.csv", index=False)
    splits['y_val'].to_csv(OUTPUT_DIR / "val_y.csv", index=False)
    splits['X_test'].to_csv(OUTPUT_DIR / "test_X.csv", index=False)
    splits['y_test'].to_csv(OUTPUT_DIR / "test_y.csv", index=False)
    
    # Save feature list
    with open(OUTPUT_DIR / "feature_list.txt", 'w') as f:
        f.write("\n".join(features))
    
    logger.info(f"Saved all processed data to {OUTPUT_DIR}")


def main():
    """Execute Phase 3."""
    logger.info("=" * 60)
    logger.info("PHASE 3: DATA PROCESSING AND FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    # Load data
    df = load_data()
    
    # Remove outliers
    df = remove_outliers(df)
    
    # Create features
    df = create_derived_features(df)
    
    # Prepare feature matrix
    X, y, features = prepare_feature_matrix(df)
    
    # Split data
    splits = split_data(X, y)
    
    # Save
    save_processed_data(df, splits, features)
    
    logger.info("\n✅ Phase 3 Complete")
    logger.info(f"   Features: {len(features)}")
    logger.info(f"   Train/Val/Test: {len(splits['X_train'])}/{len(splits['X_val'])}/{len(splits['X_test'])}")
    
    return df, splits, features


if __name__ == "__main__":
    main()
