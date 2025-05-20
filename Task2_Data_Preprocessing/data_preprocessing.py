#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing Script

This script performs the following preprocessing steps:
1. Handles missing values
2. Detects and handles outliers
3. Normalizes/scales features
4. Splits data into training and testing sets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
def load_data(file_path):
    """
    Load dataset from the given file path

    Args:
        file_path (str): Path to the dataset file

    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"Loading data from {file_path}...")

    # Determine file type based on extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    print(f"Dataset loaded successfully with shape: {df.shape}")
    return df

# Exploratory Data Analysis
def explore_data(df):
    """
    Perform basic exploratory data analysis

    Args:
        df (pandas.DataFrame): Input dataset

    Returns:
        None
    """
    print("\n--- Dataset Information ---")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Summary statistics ---")
    print(df.describe())

    print("\n--- Missing values ---")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = pd.concat([missing_values, missing_percent], axis=1)
    missing_data.columns = ['Count', 'Percent']
    print(missing_data[missing_data['Count'] > 0])

    # Create a directory for plots if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Plot histograms for numerical features
    print("\nGenerating histograms for numerical features...")
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_features) > 0:
        df[numerical_features].hist(figsize=(15, 10))
        plt.tight_layout()
        plt.savefig('plots/histograms.png')
        plt.close()

    # Plot correlation matrix
    if len(numerical_features) > 1:
        print("Generating correlation matrix...")
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numerical_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png')
        plt.close()

# Handle missing values
def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset

    Args:
        df (pandas.DataFrame): Input dataset
        strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent', 'constant')

    Returns:
        pandas.DataFrame: Dataset with handled missing values
    """
    print(f"\nHandling missing values using {strategy} strategy...")

    # Create a copy of the dataframe
    df_processed = df.copy()

    # Handle numerical and categorical features separately
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Impute numerical features
    if len(numerical_features) > 0:
        imputer = SimpleImputer(strategy=strategy)
        df_processed[numerical_features] = imputer.fit_transform(df[numerical_features])

    # Impute categorical features with most frequent value
    if len(categorical_features) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

    print("Missing values handled successfully.")
    return df_processed

# Detect and handle outliers
def handle_outliers(df, method='iqr', contamination=0.05):
    """
    Detect and handle outliers in the dataset

    Args:
        df (pandas.DataFrame): Input dataset
        method (str): Method for outlier detection ('iqr' or 'isolation_forest')
        contamination (float): Expected proportion of outliers (for isolation_forest)

    Returns:
        pandas.DataFrame: Dataset with handled outliers
    """
    print(f"\nDetecting and handling outliers using {method} method...")

    # Create a copy of the dataframe
    df_processed = df.copy()

    # Get numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

    if method == 'iqr':
        # IQR method for outlier detection
        for column in numerical_features:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Replace outliers with bounds
            df_processed[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df_processed[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

            outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            print(f"  - {column}: {outliers_count} outliers detected and handled")

    elif method == 'isolation_forest':
        # Isolation Forest for outlier detection
        if len(numerical_features) > 0:
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(df[numerical_features])

            # Mark outliers (-1) and inliers (1)
            outlier_indices = np.where(outliers == -1)[0]
            print(f"  - {len(outlier_indices)} outliers detected using Isolation Forest")

            # For demonstration, we'll just print the outlier indices
            # In a real scenario, you might want to handle these differently
            print(f"  - Outlier indices: {outlier_indices[:10]}{'...' if len(outlier_indices) > 10 else ''}")

            # Option 1: Remove outliers
            # df_processed = df_processed.drop(outlier_indices)

            # Option 2: Replace outliers with mean/median values
            for column in numerical_features:
                column_median = df[column].median()
                df_processed.loc[outlier_indices, column] = column_median

    else:
        raise ValueError("Unsupported outlier detection method. Use 'iqr' or 'isolation_forest'.")

    print("Outliers handled successfully.")
    return df_processed

# Normalize or scale features
def scale_features(df, method='standard'):
    """
    Normalize or scale features in the dataset

    Args:
        df (pandas.DataFrame): Input dataset
        method (str): Scaling method ('standard' or 'minmax')

    Returns:
        pandas.DataFrame: Dataset with scaled features
        object: Fitted scaler object
    """
    print(f"\nScaling features using {method} scaling...")

    # Create a copy of the dataframe
    df_processed = df.copy()

    # Get numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numerical_features) == 0:
        print("No numerical features to scale.")
        return df_processed, None

    # Initialize the appropriate scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaling method. Use 'standard' or 'minmax'.")

    # Fit and transform the numerical features
    df_processed[numerical_features] = scaler.fit_transform(df[numerical_features])

    print("Features scaled successfully.")
    return df_processed, scaler

# Split data into training and testing sets
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets

    Args:
        df (pandas.DataFrame): Input dataset
        target_column (str): Name of the target column
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\nSplitting data into training ({1-test_size:.0%}) and testing ({test_size:.0%}) sets...")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test

# Save processed data
def save_processed_data(X_train, X_test, y_train, y_test, output_dir='data/processed_data'):
    """
    Save processed data to files

    Args:
        X_train, X_test, y_train, y_test: Processed data splits
        output_dir (str): Directory to save the processed data

    Returns:
        None
    """
    print(f"\nSaving processed data to {output_dir} directory...")

    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save data splits to CSV files
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print("Processed data saved successfully.")

# Main function
def main():
    """
    Main function to execute the data preprocessing pipeline
    """
    print("Starting data preprocessing pipeline...\n")

    # Load the dataset
    dataset_path = "data/raw/CVD_cleaned.csv"  # Updated path to use the CVD dataset in data/raw directory
    df = load_data(dataset_path)

    # Explore the data
    explore_data(df)

    # Handle missing values
    df_no_missing = handle_missing_values(df, strategy='median')

    # Handle outliers
    df_no_outliers = handle_outliers(df_no_missing, method='iqr')

    # Scale features
    df_scaled, scaler = scale_features(df_no_outliers, method='standard')

    # Identify the target column for the CVD dataset
    target_column = "Heart_Disease"  # Using Heart_Disease as the target variable
    print(f"\nUsing '{target_column}' as the target column for cardiovascular disease prediction.")

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df_scaled, target_column)

    # Save the processed data
    save_processed_data(X_train, X_test, pd.DataFrame(y_train), pd.DataFrame(y_test))

    print("\nData preprocessing completed successfully!")

if __name__ == "__main__":
    main()
