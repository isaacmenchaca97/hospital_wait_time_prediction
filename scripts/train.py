import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_args():
    """Parse job arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    parser.add_argument("--eval_metric", type=str, default="rmse")
    parser.add_argument("--nfold", type=int, default=5)
    parser.add_argument("--early_stopping_rounds", type=int, default=10)

    # SageMaker specific arguments
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    args, _ = parser.parse_known_args()
    logger.info(f"Received arguments {args}")
    return args


def load_data(data_dir, filename):
    """
    Load data from the specified directory

    Args:
        data_dir (str): Directory containing the data file
        filename (str): Name of the data file

    Returns:
        tuple: Features DataFrame and labels Series
    """
    data = pd.read_csv(f"{data_dir}/{filename}")
    features = data.drop("tiempo_total", axis=1)
    labels = data["tiempo_total"]
    return features, labels


def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        dict: Dictionary containing regression metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def train_model(args, dtrain, dvalidation):
    """
    Train XGBoost regression model with cross-validation

    Args:
        args: Parsed command line arguments
        dtrain: Training data as DMatrix
        dvalidation: Validation data as DMatrix

    Returns:
        tuple: Trained model and metrics
    """
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": args.objective,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }

    # Run cross-validation
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        nfold=args.nfold,
        early_stopping_rounds=args.early_stopping_rounds,
        metrics=[args.eval_metric],
        seed=42,
    )

    # Train final model
    evallist = [(dtrain, "train"), (dvalidation, "validation")]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=len(cv_results),
        evals=evallist,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    # Generate predictions
    train_pred = model.predict(dtrain)
    validation_pred = model.predict(dvalidation)

    # Calculate metrics
    train_metrics = calculate_regression_metrics(dtrain.get_label(), train_pred)
    validation_metrics = calculate_regression_metrics(dvalidation.get_label(), validation_pred)

    metrics = {"train": train_metrics, "validation": validation_metrics}

    return model, metrics


def save_model_artifacts(model, metrics, args):
    """
    Save model artifacts and metrics

    Args:
        model: Trained XGBoost model
        metrics (dict): Model metrics
        args: Parsed command line arguments
    """
    metrics_data = {
        "hyperparameters": {
            "max_depth": args.max_depth,
            "eta": args.eta,
            "objective": args.objective,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        },
        "regression_metrics": {
            "validation": {
                "rmse": metrics["validation"]["rmse"],
                "mae": metrics["validation"]["mae"],
                "r2": metrics["validation"]["r2"],
            },
            "train": {
                "rmse": metrics["train"]["rmse"],
                "mae": metrics["train"]["mae"],
                "r2": metrics["train"]["r2"],
            },
        },
    }

    # Save metrics
    metrics_location = os.path.join(args.output_data_dir, "metrics.json")
    with open(metrics_location, "w") as f:
        json.dump(metrics_data, f)

    # Save model
    model_location = os.path.join(args.model_dir, "xgboost-model")
    with open(model_location, "wb") as f:
        joblib.dump(model, f)


if __name__ == "__main__":
    """Main training function"""
    args = parse_args()

    # Load data
    train_features, train_labels = load_data(args.train_data_dir, "train.csv")
    validation_features, validation_labels = load_data(args.validation_data_dir, "validation.csv")

    # Create DMatrix objects
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dvalidation = xgb.DMatrix(validation_features, label=validation_labels)

    # Train model
    model, metrics = train_model(args, dtrain, dvalidation)

    # Log metrics
    logger.info("Training metrics:")
    logger.info(f"RMSE: {metrics['train']['rmse']:.4f}")
    logger.info(f"MAE: {metrics['train']['mae']:.4f}")
    logger.info(f"R2: {metrics['train']['r2']:.4f}")

    logger.info("\nValidation metrics:")
    logger.info(f"RMSE: {metrics['validation']['rmse']:.4f}")
    logger.info(f"MAE: {metrics['validation']['mae']:.4f}")
    logger.info(f"R2: {metrics['validation']['r2']:.4f}")

    # Save artifacts
    save_model_artifacts(model, metrics, args)
