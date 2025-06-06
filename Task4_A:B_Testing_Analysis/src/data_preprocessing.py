"""
Data preprocessing module for Wine A/B Testing Analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class WineDataPreprocessor:
    """Class for preprocessing wine dataset for A/B testing analysis"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data = None
        self.processed_data = None

    def load_data(self, file_path=None):
        """Load wine dataset from CSV file"""
        if file_path is None:
            file_path = WINE_DATA_PATH

        try:
            # Load data without headers since the original dataset doesn't have them
            self.data = pd.read_csv(file_path, header=None)
            self.data.columns = WINE_FEATURES
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def explore_data(self):
        """Perform basic data exploration"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        print("=== WINE DATASET EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns)}")
        print("\nFirst 5 rows:")
        print(self.data.head())

        print("\nDataset info:")
        print(self.data.info())

        print("\nBasic statistics:")
        print(self.data.describe())

        print("\nMissing values:")
        print(self.data.isnull().sum())

        print("\nWine class distribution:")
        print(self.data['class'].value_counts().sort_index())

        return {
            'shape': self.data.shape,
            'missing_values': self.data.isnull().sum().sum(),
            'class_distribution': self.data['class'].value_counts().to_dict()
        }

    def clean_data(self):
        """Clean and prepare data for analysis"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None

        # Create a copy for processing
        self.processed_data = self.data.copy()

        # Check for missing values
        if self.processed_data.isnull().sum().sum() > 0:
            print("Handling missing values...")
            # Fill missing values with median for numerical columns
            numerical_cols = self.processed_data.select_dtypes(include=[np.number]).columns
            self.processed_data[numerical_cols] = self.processed_data[numerical_cols].fillna(
                self.processed_data[numerical_cols].median()
            )

        # Remove any duplicate rows
        initial_shape = self.processed_data.shape[0]
        self.processed_data = self.processed_data.drop_duplicates()
        final_shape = self.processed_data.shape[0]

        if initial_shape != final_shape:
            print(f"Removed {initial_shape - final_shape} duplicate rows")

        print("Data cleaning completed.")
        return self.processed_data

    def create_ab_test_groups(self, feature='alcohol', threshold_percentile=50):
        """Create A/B test groups based on a feature threshold"""
        if self.processed_data is None:
            print("No processed data available. Please clean data first.")
            return None

        # Calculate threshold
        threshold = np.percentile(self.processed_data[feature], threshold_percentile)

        # Create groups
        self.processed_data['ab_group'] = np.where(
            self.processed_data[feature] >= threshold, 'Treatment', 'Control'
        )

        # Add binary group indicator
        self.processed_data['group_binary'] = np.where(
            self.processed_data['ab_group'] == 'Treatment', 1, 0
        )

        print(f"A/B test groups created based on {feature} (threshold: {threshold:.2f})")
        print(f"Control group size: {sum(self.processed_data['ab_group'] == 'Control')}")
        print(f"Treatment group size: {sum(self.processed_data['ab_group'] == 'Treatment')}")

        return self.processed_data

    def save_processed_data(self, filename='wine_processed.csv'):
        """Save processed data to file"""
        if self.processed_data is None:
            print("No processed data to save.")
            return

        output_path = os.path.join(PROCESSED_DATA_DIR, filename)
        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

    def get_feature_statistics(self, feature):
        """Get statistics for a specific feature by group"""
        if self.processed_data is None or 'ab_group' not in self.processed_data.columns:
            print("No A/B test groups available.")
            return None

        stats = self.processed_data.groupby('ab_group')[feature].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(3)

        return stats

    def detect_outliers(self, method='iqr', contamination=0.1):
        """Detect outliers using various methods"""
        if self.processed_data is None:
            print("No processed data available.")
            return None

        numerical_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        outlier_indices = set()

        for col in numerical_cols:
            if method == 'iqr':
                Q1 = self.processed_data[col].quantile(0.25)
                Q3 = self.processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.processed_data[(self.processed_data[col] < lower_bound) |
                                             (self.processed_data[col] > upper_bound)].index
                outlier_indices.update(outliers)

            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.processed_data[col]))
                outliers = self.processed_data[z_scores > 3].index
                outlier_indices.update(outliers)

        print(f"Detected {len(outlier_indices)} outliers using {method} method")
        return list(outlier_indices)

    def advanced_feature_engineering(self):
        """Create advanced features for A/B testing"""
        if self.processed_data is None:
            print("No processed data available.")
            return None

        # Create ratio features
        self.processed_data['phenols_to_flavanoids'] = (
            self.processed_data['total_phenols'] / (self.processed_data['flavanoids'] + 1e-8)
        )

        self.processed_data['alcohol_to_acid'] = (
            self.processed_data['alcohol'] / (self.processed_data['malic_acid'] + 1e-8)
        )

        # Create interaction features
        self.processed_data['alcohol_x_phenols'] = (
            self.processed_data['alcohol'] * self.processed_data['total_phenols']
        )

        # Create binned features
        self.processed_data['alcohol_category'] = pd.cut(
            self.processed_data['alcohol'],
            bins=3,
            labels=['Low', 'Medium', 'High']
        )

        self.processed_data['phenols_category'] = pd.cut(
            self.processed_data['total_phenols'],
            bins=3,
            labels=['Low', 'Medium', 'High']
        )

        # Create quality score (composite metric)
        numerical_features = ['alcohol', 'total_phenols', 'flavanoids', 'color_intensity', 'hue']

        # Normalize features for quality score
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(self.processed_data[numerical_features])

        # Simple quality score (weighted sum)
        weights = [0.3, 0.25, 0.25, 0.15, 0.05]  # Based on wine quality importance
        self.processed_data['quality_score'] = np.dot(normalized_features, weights)

        print("Advanced feature engineering completed.")
        print(f"New features created: {self.processed_data.shape[1] - len(WINE_FEATURES)} additional features")

        return self.processed_data

    def create_stratified_groups(self, feature='alcohol', n_strata=3):
        """Create stratified A/B test groups"""
        if self.processed_data is None:
            print("No processed data available.")
            return None

        # Create strata based on feature quantiles
        self.processed_data['stratum'] = pd.qcut(
            self.processed_data[feature],
            q=n_strata,
            labels=[f'Stratum_{i+1}' for i in range(n_strata)]
        )

        # Assign A/B groups within each stratum
        np.random.seed(RANDOM_SEED)
        self.processed_data['ab_group_stratified'] = ''

        for stratum in self.processed_data['stratum'].unique():
            stratum_data = self.processed_data[self.processed_data['stratum'] == stratum]
            n_stratum = len(stratum_data)

            # Random assignment within stratum
            assignments = ['Control'] * (n_stratum // 2) + ['Treatment'] * (n_stratum - n_stratum // 2)
            np.random.shuffle(assignments)

            self.processed_data.loc[stratum_data.index, 'ab_group_stratified'] = assignments

        print(f"Stratified A/B groups created with {n_strata} strata")
        print("Group distribution by stratum:")
        print(self.processed_data.groupby(['stratum', 'ab_group_stratified']).size().unstack())

        return self.processed_data

    def perform_dimensionality_reduction(self, n_components=2, method='pca'):
        """Perform dimensionality reduction for visualization"""
        if self.processed_data is None:
            print("No processed data available.")
            return None

        # Select numerical features
        numerical_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['class', 'ab_group', 'group_binary']]

        X = self.processed_data[numerical_cols].fillna(0)

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)
            X_reduced = reducer.fit_transform(StandardScaler().fit_transform(X))

            # Add explained variance info
            explained_variance = reducer.explained_variance_ratio_
            print(f"PCA completed. Explained variance: {explained_variance}")

            # Add components to dataframe
            for i in range(n_components):
                self.processed_data[f'PC{i+1}'] = X_reduced[:, i]

        return self.processed_data

    def generate_data_quality_report(self):
        """Generate comprehensive data quality report"""
        if self.data is None:
            print("No data loaded.")
            return None

        report = {
            'basic_info': {
                'shape': self.data.shape,
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'dtypes': self.data.dtypes.value_counts().to_dict()
            },
            'missing_data': {
                'total_missing': self.data.isnull().sum().sum(),
                'missing_by_column': self.data.isnull().sum().to_dict(),
                'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
            },
            'duplicates': {
                'total_duplicates': self.data.duplicated().sum(),
                'duplicate_percentage': self.data.duplicated().sum() / len(self.data) * 100
            },
            'numerical_summary': {},
            'categorical_summary': {}
        }

        # Numerical features analysis
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            report['numerical_summary'][col] = {
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'skewness': self.data[col].skew(),
                'kurtosis': self.data[col].kurtosis(),
                'zeros': (self.data[col] == 0).sum(),
                'unique_values': self.data[col].nunique()
            }

        # Categorical features analysis
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            report['categorical_summary'][col] = {
                'unique_values': self.data[col].nunique(),
                'most_frequent': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                'value_counts': self.data[col].value_counts().to_dict()
            }

        return report

if __name__ == "__main__":
    # Example usage
    preprocessor = WineDataPreprocessor()

    # Load and explore data
    data = preprocessor.load_data()
    if data is not None:
        exploration_results = preprocessor.explore_data()

        # Clean data
        cleaned_data = preprocessor.clean_data()

        # Create A/B test groups
        ab_data = preprocessor.create_ab_test_groups(feature='alcohol')

        # Save processed data
        preprocessor.save_processed_data()

        print("\nData preprocessing completed successfully!")
