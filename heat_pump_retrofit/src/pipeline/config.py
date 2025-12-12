#!/usr/bin/env python3
"""
Configuration file for Heat Pump Retrofit Analysis Pipeline
All parameters, paths, and constants in one place for reproducibility.
"""

from pathlib import Path
import numpy as np

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "figures_revised"
TABLES_DIR = PROJECT_ROOT / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for d in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# =============================================================================
# RANDOM SEED (for reproducibility)
# =============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# DATA PARAMETERS
# =============================================================================
# RECS 2020 filtering
GAS_HEATING_CODE = 1  # FUELHEAT == 1 for natural gas
OUTLIER_PERCENTILES = (2, 98)  # Remove 2-98 percentile outliers

# Train/Val/Test split ratios
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
# Raw features from RECS
RAW_FEATURES = [
    'HDD65', 'CDD65', 'TOTCSQFT', 'TOTSQFT_EN', 'YEARMADERANGE',
    'TYPEHUQ', 'STORIES', 'NCOMBATH', 'NHSLDMEM', 'MONESSION',
    'ADQINSUL', 'DRAFTY', 'WINDOWS', 'HIGHCEIL', 'WALLTYPE',
    'ROOFTYPE', 'FUELHEAT', 'EQUIPM', 'DIVISION', 'REGIONC', 'NWEIGHT'
]

# Derived features
DERIVED_FEATURES = [
    'log_sqft', 'building_age', 'envelope_score', 'hdd_sqft',
    'age_hdd', 'sqft_per_hdd', 'hdd_squared'
]

# Target variable
TARGET = 'thermal_intensity'  # BTU/sqft/HDD

# Envelope class definitions
ENVELOPE_CLASSES = {
    'Poor': (0, 0.4),
    'Medium': (0.4, 0.7),
    'Good': (0.7, 1.0)
}

# Climate zones (HDD thresholds)
CLIMATE_ZONES = {
    'Mild': (0, 4000),
    'Moderate': (4000, 5500),
    'Cold': (5500, 12000)
}

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# Random Forest hyperparameters (for comparison)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# Cross-validation
CV_FOLDS = 5

# =============================================================================
# ECONOMIC PARAMETERS
# =============================================================================
# Discount rate and analysis horizon
DISCOUNT_RATE = 0.05  # 5%
ANALYSIS_HORIZON = 15  # years
SYSTEM_LIFETIME = 20  # years

# Energy prices ($/unit) - baseline 2024
ENERGY_PRICES = {
    'electricity': {  # $/kWh
        'low': 0.08,
        'base': 0.14,
        'high': 0.22
    },
    'natural_gas': {  # $/therm
        'low': 0.80,
        'base': 1.10,
        'high': 1.50
    }
}

# Heat pump parameters
HEAT_PUMP = {
    'COP_rated': 3.0,  # At 47°F
    'COP_17F': 2.3,    # At 17°F (cold climate)
    'HSPF': 10.0,      # Heating Seasonal Performance Factor
    'installation_cost': {  # $
        'standard': 12000,
        'cold_climate': 15000
    },
    'lifetime': 15  # years
}

# Retrofit costs ($)
RETROFIT_COSTS = {
    'none': 0,
    'air_sealing': 2500,
    'insulation_attic': 3500,
    'insulation_walls': 8000,
    'windows': 12000,
    'comprehensive': 20000
}

# Carbon intensity (kg CO2/unit)
CARBON_INTENSITY = {
    'natural_gas': 5.3,  # kg CO2/therm
    'electricity_2023': 0.386,  # kg CO2/kWh (US average)
    'electricity_2035': 0.250,  # projected
    'electricity_2050': 0.100   # projected (ambitious)
}

# =============================================================================
# MONTE CARLO PARAMETERS
# =============================================================================
MC_SAMPLES = 10000

# Parameter distributions for Monte Carlo
MC_DISTRIBUTIONS = {
    'elec_price': {'dist': 'triangular', 'params': (0.08, 0.14, 0.22)},
    'gas_price': {'dist': 'normal', 'params': (1.10, 0.25)},
    'COP': {'dist': 'truncnorm', 'params': (2.8, 0.4, 2.0, 4.5)},
    'hp_cost': {'dist': 'lognormal', 'params': (9.2, 0.3)},
    'discount_rate': {'dist': 'uniform', 'params': (0.03, 0.08)},
    'system_life': {'dist': 'discrete', 'params': ([12, 15, 18, 20], [0.15, 0.45, 0.30, 0.10])}
}

# =============================================================================
# SOBOL SENSITIVITY PARAMETERS
# =============================================================================
SOBOL_PARAMS = {
    'num_vars': 8,
    'names': ['Elec_Price', 'Gas_Price', 'HDD', 'COP', 'HP_Cost', 
              'Retrofit_Cost', 'Discount_Rate', 'System_Life'],
    'bounds': [
        [0.08, 0.22],    # Elec price $/kWh
        [0.80, 1.50],    # Gas price $/therm
        [2000, 8000],    # HDD65
        [2.0, 4.0],      # COP
        [8000, 20000],   # HP cost $
        [0, 20000],      # Retrofit cost $
        [0.03, 0.08],    # Discount rate
        [12, 20]         # System life years
    ]
}
SOBOL_SAMPLES = 1024

# =============================================================================
# VIABILITY SCORE PARAMETERS
# =============================================================================
# V = (1 - α*H*) × (1 - β*P*) × γ
VIABILITY_PARAMS = {
    'alpha': 0.59,  # HDD sensitivity
    'beta': 0.79,   # Price sensitivity
    'gamma': {      # Envelope adjustment
        'Poor': 1.00,
        'Medium': 0.74,
        'Good': 0.49
    }
}
VIABILITY_THRESHOLD = 0.5

# =============================================================================
# CENSUS DIVISIONS
# =============================================================================
CENSUS_DIVISIONS = {
    1: {'name': 'New England', 'abbr': 'NE', 'states': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT']},
    2: {'name': 'Middle Atlantic', 'abbr': 'MA', 'states': ['NJ', 'NY', 'PA']},
    3: {'name': 'East North Central', 'abbr': 'ENC', 'states': ['IL', 'IN', 'MI', 'OH', 'WI']},
    4: {'name': 'West North Central', 'abbr': 'WNC', 'states': ['IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD']},
    5: {'name': 'South Atlantic', 'abbr': 'SA', 'states': ['DE', 'DC', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV']},
    6: {'name': 'East South Central', 'abbr': 'ESC', 'states': ['AL', 'KY', 'MS', 'TN']},
    7: {'name': 'West South Central', 'abbr': 'WSC', 'states': ['AR', 'LA', 'OK', 'TX']},
    8: {'name': 'Mountain North', 'abbr': 'MtN', 'states': ['CO', 'ID', 'MT', 'UT', 'WY']},
    9: {'name': 'Mountain South', 'abbr': 'MtS', 'states': ['AZ', 'NV', 'NM']},
    10: {'name': 'Pacific', 'abbr': 'PAC', 'states': ['AK', 'CA', 'HI', 'OR', 'WA']}
}

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================
FIGURE_DPI = 300
FIGURE_FORMAT = ['png', 'pdf']
COLOR_PALETTE = {
    'poor': '#d62728',
    'medium': '#ff7f0e',
    'good': '#2ca02c',
    'cold': '#1f77b4',
    'moderate': '#9467bd',
    'mild': '#e377c2',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c'
}
COLORMAP_CONTINUOUS = 'viridis'
COLORMAP_DIVERGING = 'RdBu_r'

# =============================================================================
# APPLIED ENERGY JOURNAL REQUIREMENTS
# =============================================================================
JOURNAL_REQUIREMENTS = {
    'max_words': 8000,
    'max_figures': 15,
    'figure_formats': ['TIFF', 'EPS', 'PDF'],
    'min_dpi': 300,
    'reference_style': 'Elsevier Harvard'
}

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
