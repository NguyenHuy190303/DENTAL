"""
Configuration settings for dental health research analysis.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
BRFSS_SAS_FILE = DATA_DIR / "llcp2022.sas7bdat"
BRFSS_PARQUET_FILE = DATA_DIR / "llcp2022.parquet"
BRFSS_CLEANED_FILE = DATA_DIR / "llcp2022_cleaned.parquet"

# Analysis settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model settings
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    },
    'logistic_regression': {
        'random_state': RANDOM_STATE,
        'max_iter': 1000
    }
}

# Visualization settings
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'save_format': 'png',
    'bbox_inches': 'tight',
    'style': 'default'
}

# Color palette for health equity analysis
EQUITY_COLORS = {
    'high_risk': '#d62728',
    'medium_risk': '#ff7f0e', 
    'low_risk': '#2ca02c',
    'overall': '#1f77b4'
}

# Dental health variables mapping
DENTAL_VARIABLES = {
    'RMVTETH4': 'teeth_removed',
    'EDENTUL2': 'edentulism',
    # Thêm các biến khác khi cần
}

# Demographic variables
DEMOGRAPHIC_VARIABLES = {
    'AGE': 'age',
    'SEX': 'sex',
    'RACE': 'race',
    'INCOME2': 'income',
    'EDUCA': 'education',
    # Thêm các biến khác khi cần
}

# Health behavior variables
HEALTH_BEHAVIOR_VARIABLES = {
    'SMOKE100': 'smoking_history',
    'DRNKANY5': 'alcohol_consumption',
    'EXERANY2': 'physical_activity',
    # Thêm các biến khác khi cần
}

# Missing value codes (BRFSS specific)
MISSING_VALUE_CODES = {
    7: 'dont_know_not_sure',
    9: 'refused',
    77: 'dont_know_not_sure',
    99: 'refused',
    777: 'dont_know_not_sure',
    999: 'refused'
}
