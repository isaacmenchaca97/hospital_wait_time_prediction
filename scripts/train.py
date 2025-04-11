import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
import xgboost as xgb


def parse_args():
    """Parse job arguments"""
    parser = argparse.ArgumentParser()

    # Hyperparameters for tuning
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--eval_metric", type=str, default="auc")
    parser.add_argument("--nfold", type=int, default=3)
    parser.add_argument("--early_stopping_rounds", type=int, default=3)

    # SageMaker specific arguments
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )

    return parser.parse_args()


def load_data(data_dir, filename="data.csv"):
    """
    Load data from the specified directory

    Args:
        data_dir (str): Directory containing the data file
        filename (str): Name of the data file

    Returns:
        tuple: Features DataFrame and labels Series
    """
    data = pd.read_csv(f"{data_dir}/{filename}")
    features = data.drop("wait_time", axis=1)
    labels = pd.DataFrame(data["wait_time"])
    return features, labels


def train_model(args, dtrain, dvalidation):
    """
    Train XGBoost model with cross-validation

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
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))

    # Generate predictions
    train_pred = model.predict(dtrain)
    validation_pred = model.predict(dvalidation)

    # Calculate metrics
    train_auc = roc_auc_score(dtrain.get_label(), train_pred)
    validation_auc = roc_auc_score(dvalidation.get_label(), validation_pred)

    metrics = {"train_auc": train_auc, "validation_auc": validation_auc}

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
        "binary_classification_metrics": {
            "validation:auc": {"value": metrics["validation_auc"]},
            "train:auc": {"value": metrics["train_auc"]},
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


def main():
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
    print(f"[0]#011train-auc:{metrics['train_auc']:.2f}")
    print(f"[0]#011validation-auc:{metrics['validation_auc']:.2f}")

    # Save artifacts
    save_model_artifacts(model, metrics, args)


if __name__ == "__main__":
    main()
