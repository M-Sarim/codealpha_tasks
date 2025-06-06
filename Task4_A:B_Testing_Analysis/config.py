"""
Configuration file for A/B Testing Wine Analysis
"""

import os

# Data paths
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
WINE_DATA_PATH = os.path.join(RAW_DATA_DIR, "wine.csv")

# Analysis parameters
RANDOM_SEED = 42
ALPHA = 0.05  # Significance level
POWER = 0.8   # Statistical power
EFFECT_SIZE = 0.5  # Cohen's d for medium effect size

# A/B Test simulation parameters
CONTROL_GROUP_SIZE = 500
TREATMENT_GROUP_SIZE = 500
SIMULATION_RUNS = 1000

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
STYLE = 'seaborn-v0_8'

# Wine feature names (based on UCI Wine dataset)
WINE_FEATURES = [
    'class',
    'alcohol',
    'malic_acid',
    'ash',
    'alcalinity_of_ash',
    'magnesium',
    'total_phenols',
    'flavanoids',
    'nonflavanoid_phenols',
    'proanthocyanins',
    'color_intensity',
    'hue',
    'od280_od315_of_diluted_wines',
    'proline'
]

# A/B Test scenarios for wine analysis
AB_TEST_SCENARIOS = {
    'alcohol_content': {
        'description': 'Test impact of alcohol content on wine quality perception',
        'metric': 'alcohol',
        'hypothesis': 'Higher alcohol content wines are perceived as higher quality'
    },
    'phenol_levels': {
        'description': 'Test impact of phenol levels on consumer preference',
        'metric': 'total_phenols',
        'hypothesis': 'Wines with higher phenol levels have better consumer ratings'
    },
    'color_intensity': {
        'description': 'Test impact of color intensity on purchase decisions',
        'metric': 'color_intensity',
        'hypothesis': 'Wines with higher color intensity have higher purchase rates'
    }
}

# Advanced analysis settings
BAYESIAN_SETTINGS = {
    'prior_alpha': 1,
    'prior_beta': 1,
    'n_samples': 100000,
    'credible_interval': 0.95
}

BOOTSTRAP_SETTINGS = {
    'n_bootstrap': 10000,
    'confidence_interval': 0.95,
    'random_state': RANDOM_SEED
}

SEQUENTIAL_SETTINGS = {
    'max_looks': 5,
    'alpha_spending_function': 'obrien_fleming',  # or 'pocock'
    'min_sample_fraction': 0.2
}

MULTIPLE_TESTING_SETTINGS = {
    'correction_methods': ['bonferroni', 'holm', 'fdr_bh', 'fdr_by'],
    'default_method': 'bonferroni'
}

# Simulation settings
SIMULATION_SETTINGS = {
    'n_simulations': 1000,
    'effect_sizes': [0.0, 0.2, 0.5, 0.8, 1.0],
    'sample_sizes': [50, 100, 200, 500, 1000],
    'alpha_levels': [0.01, 0.05, 0.10]
}

# Feature engineering settings
FEATURE_ENGINEERING = {
    'create_ratios': True,
    'create_interactions': True,
    'create_bins': True,
    'create_quality_score': True,
    'outlier_detection': True,
    'dimensionality_reduction': True
}

# Outlier detection settings
OUTLIER_SETTINGS = {
    'methods': ['iqr', 'zscore', 'isolation_forest'],
    'default_method': 'iqr',
    'contamination': 0.1,
    'zscore_threshold': 3
}

# Stratification settings
STRATIFICATION_SETTINGS = {
    'n_strata': 3,
    'stratification_features': ['alcohol', 'total_phenols', 'color_intensity'],
    'balance_tolerance': 0.1
}

# Advanced visualization settings
ADVANCED_VIZ_SETTINGS = {
    'plot_types': [
        'bayesian_posterior',
        'bootstrap_distribution',
        'sequential_monitoring',
        'multiple_testing_correction',
        'power_curves',
        'effect_size_forest_plot'
    ],
    'interactive_plots': True,
    'save_formats': ['png', 'pdf', 'svg'],
    'high_dpi': True
}

# Reporting settings
REPORTING_SETTINGS = {
    'include_executive_summary': True,
    'include_technical_details': True,
    'include_recommendations': True,
    'include_limitations': True,
    'export_formats': ['csv', 'excel', 'json'],
    'auto_generate_report': True
}
