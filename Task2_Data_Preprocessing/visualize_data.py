#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Visualization Script

This script creates professional visualizations for the dataset before and after preprocessing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
colors = sns.color_palette("viridis", 8)

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    """Load original and processed datasets"""
    # Load original data
    original_data = pd.read_csv('data/raw/CVD_cleaned.csv')

    # Load processed data
    X_train = pd.read_csv('data/processed_data/X_train.csv')
    X_test = pd.read_csv('data/processed_data/X_test.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv')
    y_test = pd.read_csv('data/processed_data/y_test.csv')

    return original_data, X_train, X_test, y_train, y_test

def visualize_data_distribution(data, output_dir='plots'):
    """Create distribution plots for numerical features"""
    create_directory(output_dir)

    # Create a copy of the data
    data_numeric = data.copy()

    # Select numerical columns
    numerical_cols = data_numeric.select_dtypes(include=['int64', 'float64']).columns

    # Create distribution plots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols):
        plt.subplot(3, 3, i+1)
        sns.histplot(data[col], kde=True, color=colors[i])
        plt.title(f'Distribution of {col}')
        plt.tight_layout()

    plt.savefig(f'{output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create boxplots for numerical features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols):
        plt.subplot(3, 3, i+1)
        sns.boxplot(y=data[col], color=colors[i])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()

    plt.savefig(f'{output_dir}/feature_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_correlation_matrix(data, output_dir='plots'):
    """Create correlation matrix heatmap"""
    create_directory(output_dir)

    # Create a copy of the data
    data_numeric = data.copy()

    # Select numerical columns
    numerical_cols = data_numeric.select_dtypes(include=['int64', 'float64']).columns

    # Create correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = data_numeric[numerical_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                linewidths=0.5, vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_pca(X_train, y_train, output_dir='plots'):
    """Create PCA visualization"""
    create_directory(output_dir)

    # Get only numerical features for PCA
    X_train_numeric = X_train.select_dtypes(include=['int64', 'float64'])

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_numeric)

    # Create PCA plot
    plt.figure(figsize=(10, 8))

    # Create a simple scatter plot without coloring by target
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, edgecolors='w')

    # Add title and labels
    plt.title('PCA: 2D Projection of the Dataset', fontsize=18)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')

    # Add a note about the target variable
    plt.figtext(0.5, 0.01, 'Note: Points represent samples projected onto the first two principal components.',
               ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create scree plot (explained variance)
    pca_full = PCA()
    pca_full.fit(X_train_numeric)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
             np.cumsum(pca_full.explained_variance_ratio_),
             marker='o', linestyle='-', color=colors[0])
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    plt.title('Cumulative Explained Variance by PCA Components', fontsize=18)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_explained_variance.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_importance(X_train, y_train, output_dir='plots'):
    """Create feature importance visualization using a Random Forest model"""
    create_directory(output_dir)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # Get only numerical features
    X_train_numeric = X_train.select_dtypes(include=['int64', 'float64'])

    # Check if y_train is a DataFrame or Series
    if hasattr(y_train, 'iloc'):
        # Get the first value to check its type
        first_val = y_train.iloc[0]
    else:
        first_val = y_train[0]

    # Check if the target is categorical
    if isinstance(first_val, str):
        le = LabelEncoder()
        y_values = y_train.values.ravel() if hasattr(y_train, 'values') else np.array(y_train).ravel()
        y_encoded = le.fit_transform(y_values)
    else:
        y_encoded = y_train.values.ravel() if hasattr(y_train, 'values') else np.array(y_train).ravel()

    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_numeric, y_encoded)

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    plt.bar(range(X_train_numeric.shape[1]), importances[indices], align='center', color=colors)
    plt.xticks(range(X_train_numeric.shape[1]), X_train_numeric.columns[indices], rotation=90)
    plt.title('Feature Importance from Random Forest', fontsize=18)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_before_after_scaling(original_data, processed_data, output_dir='plots'):
    """Create before/after scaling comparison plots"""
    create_directory(output_dir)

    # Select numerical columns
    numerical_cols = original_data.select_dtypes(include=['int64', 'float64']).columns

    # Create before/after scaling plots for each feature
    for i, col in enumerate(numerical_cols):
        if i >= len(colors):  # Ensure we don't exceed the color palette
            i = i % len(colors)

        plt.figure(figsize=(12, 6))

        # Before scaling
        plt.subplot(1, 2, 1)
        sns.histplot(original_data[col], kde=True, color=colors[i])
        plt.title(f'{col} (Before Scaling)')

        # After scaling (only if column exists in processed data)
        plt.subplot(1, 2, 2)
        if col in processed_data.columns:
            sns.histplot(processed_data[col], kde=True, color=colors[i])
            plt.title(f'{col} (After Scaling)')
        else:
            plt.text(0.5, 0.5, f"Column '{col}' not in processed data",
                    ha='center', va='center', fontsize=12)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/scaling_comparison_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to execute the visualization pipeline"""
    print("Starting data visualization pipeline...\n")

    # Load data
    original_data, X_train, X_test, y_train, y_test = load_data()

    # Create visualizations
    print("Generating feature distribution plots...")
    visualize_data_distribution(original_data)

    print("Generating correlation matrix...")
    visualize_correlation_matrix(original_data)

    print("Generating PCA visualization...")
    visualize_pca(X_train, y_train)

    print("Generating feature importance visualization...")
    visualize_feature_importance(X_train, y_train)

    print("Generating before/after scaling comparison plots...")
    visualize_before_after_scaling(original_data, X_train)

    print("\nData visualization completed successfully!")
    print(f"All visualizations saved to the 'plots' directory.")

if __name__ == "__main__":
    main()
