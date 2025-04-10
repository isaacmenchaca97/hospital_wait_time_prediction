import argparse
from datetime import datetime
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args():
    """Parse job arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-data", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def load_data(input_path):
    """Load data from input path"""
    df = pd.read_csv(os.path.join(input_path, "data.csv"))
    return df


def clean_duplicates(df):
    """Remove duplicate records"""
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_len - len(df)} duplicate records")
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Calculate missing values percentage
    missing_values = df.isnull().sum() / len(df) * 100

    for column in df.columns:
        missing_pct = missing_values[column]
        if missing_pct > 0:
            if missing_pct > 30:
                # Drop columns with more than 30% missing values
                df = df.drop(columns=[column])
                print(f"Dropped column {column} with {missing_pct:.2f}% missing values")
            else:
                # For numerical columns, fill with median
                if df[column].dtype in ["int64", "float64"]:
                    df[column] = df[column].fillna(df[column].median())
                    print(f"Filled numerical missing values in {column} with median")
                # For categorical columns, fill with mode
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])
                    print(f"Filled categorical missing values in {column} with mode")

    return df


def detect_outliers(df, columns, n_std=3):
    """
    Detect outliers using the z-score method
    Returns a boolean mask of outliers
    """
    outliers = np.zeros(len(df), dtype=bool)
    for column in columns:
        if df[column].dtype in ["int64", "float64"]:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            column_outliers = z_scores > n_std
            outliers = outliers | column_outliers
            print(f"Found {column_outliers.sum()} outliers in {column}")

    return outliers


def process_dates(df):
    """Extract features from date columns"""
    date_columns = df.select_dtypes(include=["datetime64"]).columns

    for column in date_columns:
        df[f"{column}_year"] = df[column].dt.year
        df[f"{column}_month"] = df[column].dt.month
        df[f"{column}_day"] = df[column].dt.day
        df[f"{column}_hour"] = df[column].dt.hour
        df[f"{column}_dayofweek"] = df[column].dt.dayofweek

        print(f"Extracted time features from {column}")

    return df


def encode_categorical(df):
    """Encode categorical variables"""
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        # For binary categories, use simple label encoding
        if df[column].nunique() == 2:
            df[f"{column}_encoded"] = pd.factorize(df[column])[0]
        # For multiple categories, use one-hot encoding
        else:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[column])

        print(f"Encoded categorical variable: {column}")

    return df


def scale_numerical(df):
    """Scale numerical features"""
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()

    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    print(f"Scaled {len(numerical_columns)} numerical features")

    return df


def main():
    """Main preprocessing function"""
    args = parse_args()

    print("Starting preprocessing job...")

    # Load data
    df = load_data(args.input_data)
    print(f"Loaded dataset with shape: {df.shape}")

    # Clean duplicates
    df = clean_duplicates(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Convert date columns to datetime
    date_columns = ["fecha_ingreso", "fecha_egreso"]  # Add your actual date columns
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Process dates
    df = process_dates(df)

    # Detect outliers
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    outliers = detect_outliers(df, numerical_columns)
    df["is_outlier"] = outliers

    # Encode categorical variables
    df = encode_categorical(df)

    # Scale numerical features
    df = scale_numerical(df)

    # Save processed data
    output_path = os.path.join(args.output_data, "processed_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")

    # Save processing summary
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "outliers_detected": outliers.sum(),
        "processing_timestamp": datetime.now().isoformat(),
    }

    summary_path = os.path.join(args.output_data, "processing_summary.json")
    pd.Series(summary).to_json(summary_path)
    print(f"Saved processing summary to: {summary_path}")


if __name__ == "__main__":
    main()
