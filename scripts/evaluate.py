import json
import logging
import pathlib
import pickle
import tarfile

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import xgboost as xgb

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def load_model():
    """Load model from training script"""
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    # The name of the file should match how the model was saved in the training script
    model = pickle.load(open("xgboost-model", "rb"))
    return model


def load_data():
    """Loada data from input path"""
    logger.debug("Reading test data.")
    input_path = "/opt/ml/processing/test/test.csv"
    df_test = pd.read_csv(input_path)

    # Extract test set target column
    y_test = df_test["tiempo_total"]

    # Extract test set feature columns
    X = df_test.drop("tiempo_total", axis=1)
    x_test = xgb.DMatrix(X)
    return x_test, y_test


def calculate_regression_metrics(x_test, y_test, model):
    """Calculate regression metrics"""
    logger.info("Generating predictions for test data.")
    pred = model.predict(x_test)

    # Calculate model evaluation score
    logger.debug("Calculating regression metrics.")
    rmse = root_mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    metric_dict = {"regression_metrics": {"test": {"rmse": rmse, "mae": mae, "r2": r2}}}
    return metric_dict


def save_model_evaluation(metric_dict):
    """Save model evaluation metrics"""
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing evaluation report with regression metrics: {metric_dict}")
    evaluation_location = f"{output_dir}/evaluation.json"
    with open(evaluation_location, "w") as f:
        json.dump(metric_dict, f)


if __name__ == "__main__":
    """Main evaluate function"""
    # Load model
    model = load_data()

    # Load data
    x_test, y_test = load_data()

    # Calculate regression metric
    test_metrics = calculate_regression_metrics(x_test, y_test, model)

    # Save model evaluation metrics
    save_model_evaluation(test_metrics)
