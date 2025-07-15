"""
Common imports for dental health research analysis.
Centralizes frequently used imports to reduce duplication.
"""

# Standard library
import warnings
warnings.filterwarnings('ignore')

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical analysis
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Machine Learning - Preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Machine Learning - Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

# Machine Learning - Evaluation
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    recall_score, precision_score, f1_score, accuracy_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Advanced analysis
import shap

# Configuration for plots
plt.style.use('default')
sns.set_palette("husl")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Common plot settings
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'save_format': 'png',
    'bbox_inches': 'tight'
}

# Color palette for health equity analysis
EQUITY_COLORS = {
    'high_risk': '#d62728',
    'medium_risk': '#ff7f0e', 
    'low_risk': '#2ca02c',
    'overall': '#1f77b4'
}
