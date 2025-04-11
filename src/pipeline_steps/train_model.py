import boto3
import pandas as pd
import sagemaker
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
)
from sagemaker.deserializers import CSVDeserializer
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter
from sagemaker.xgboost.estimator import XGBoost


class TrainModel:
    def __init__(self, role_arn, bucket_name, region_name="us-east-1"):
        """
        Initialize the XGBoost trainer with AWS credentials and configurations.

        Args:
            role_arn (str): AWS IAM role ARN with necessary permissions
            bucket_name (str): S3 bucket name for storing data
            region_name (str): AWS region name
        """
        self.role = role_arn
        self.bucket = bucket_name
        self.region = region_name
        self.sagemaker_session = sagemaker.Session()
        self.s3_client = boto3.client("s3")

        # Set default configurations
        self.train_instance_count = 1
        self.train_instance_type = "ml.m4.xlarge"
        self.predictor_instance_count = 1
        self.predictor_instance_type = "ml.m4.xlarge"
        self.clarify_instance_count = 1
        self.clarify_instance_type = "ml.m4.xlarge"

        # Set model naming
        self.model_name = "hospital-wait-time-xgb-model"
        self.endpoint_name_prefix = "xgb-hospital-model-dev"

        # Set S3 paths
        self.output_path = f"s3://{self.bucket}/model-output"
        self.training_job_prefix = "xgb-hospital-training"

    def setup_training_job(self):
        """Set up the XGBoost training job with default hyperparameters"""
        self.static_hyperparameters = {
            "eval_metric": "auc",
            "objective": "binary:logistic",
            "num_round": "5",
        }

        self.xgb_estimator = XGBoost(
            entry_point="training_script.py",
            output_path=self.output_path,
            code_location=self.output_path,
            hyperparameters=self.static_hyperparameters,
            role=self.role,
            instance_count=self.train_instance_count,
            instance_type=self.train_instance_type,
            framework_version="1.3-1",
            base_job_name=self.training_job_prefix,
        )

    def setup_hyperparameter_tuning(self):
        """Set up hyperparameter tuning job configuration"""
        self.hyperparameter_ranges = {
            "eta": ContinuousParameter(0, 1),
            "subsample": ContinuousParameter(0.7, 0.95),
            "colsample_bytree": ContinuousParameter(0.7, 0.95),
            "max_depth": IntegerParameter(1, 5),
        }

        self.tuner = HyperparameterTuner(
            estimator=self.xgb_estimator,
            objective_metric_name="validation:auc",
            hyperparameter_ranges=self.hyperparameter_ranges,
            max_jobs=5,
            max_parallel_jobs=2,
            base_tuning_job_name="xgb-hospital-tune",
            strategy="Random",
        )

    def train_model(self, train_data_path, validation_data_path):
        """
        Train the XGBoost model with hyperparameter tuning

        Args:
            train_data_path (str): S3 path to training data
            validation_data_path (str): S3 path to validation data
        """
        s3_input_train = TrainingInput(
            s3_data=train_data_path, content_type="csv", s3_data_type="S3Prefix"
        )
        s3_input_validation = TrainingInput(
            s3_data=validation_data_path, content_type="csv", s3_data_type="S3Prefix"
        )

        self.tuner.fit(
            inputs={"train": s3_input_train, "validation": s3_input_validation},
            include_cls_metadata=False,
        )

        # Wait for tuning job to complete
        self.tuner.wait()

    def analyze_results(self):
        """Analyze hyperparameter tuning results"""
        results_df = sagemaker.HyperparameterTuningJobAnalytics(
            self.tuner.latest_tuning_job.job_name
        ).dataframe()

        # Filter and sort results
        results_df = results_df[results_df["FinalObjectiveValue"] > -float("inf")].sort_values(
            "FinalObjectiveValue", ascending=False
        )

        return results_df

    def setup_bias_analysis(self, train_data_uri, train_df_cols):
        """
        Set up bias analysis configuration

        Args:
            train_data_uri (str): URI to training data
            train_df_cols (list): List of column names
        """
        self.clarify_processor = sagemaker.clarify.SageMakerClarifyProcessor(
            role=self.role,
            instance_count=self.clarify_instance_count,
            instance_type=self.clarify_instance_type,
            sagemaker_session=self.sagemaker_session,
        )

        self.bias_data_config = DataConfig(
            s3_data_input_path=train_data_uri,
            s3_output_path=f"s3://{self.bucket}/clarify-output/bias",
            label="wait_time",
            headers=train_df_cols,
            dataset_type="text/csv",
        )

        self.model_config = ModelConfig(
            model_name=self.model_name,
            instance_type=self.train_instance_type,
            instance_count=1,
            accept_type="text/csv",
        )

        self.predictions_config = ModelPredictedLabelConfig(probability_threshold=0.5)

        self.bias_config = BiasConfig(
            label_values_or_threshold=[0],
            facet_name="patient_gender",
            facet_values_or_threshold=[1],
        )

    def run_bias_analysis(self):
        """Run bias analysis using SageMaker Clarify"""
        self.clarify_processor.run_bias(
            data_config=self.bias_data_config,
            bias_config=self.bias_config,
            model_config=self.model_config,
            model_predicted_label_config=self.predictions_config,
            pre_training_methods=["CI"],
            post_training_methods=["DPPL"],
        )

    def setup_model_explainability(self, train_data_uri, train_df_cols, baseline_data):
        """
        Set up model explainability analysis

        Args:
            train_data_uri (str): URI to training data
            train_df_cols (list): List of column names
            baseline_data (list): Baseline data for SHAP analysis
        """
        self.explainability_data_config = DataConfig(
            s3_data_input_path=train_data_uri,
            s3_output_path=f"s3://{self.bucket}/clarify-output/explainability",
            label="wait_time",
            headers=train_df_cols,
            dataset_type="text/csv",
        )

        self.shap_config = SHAPConfig(
            baseline=baseline_data,
            num_samples=500,
            agg_method="mean_abs",
            save_local_shap_values=True,
        )

    def run_explainability_analysis(self):
        """Run model explainability analysis"""
        self.clarify_processor.run_explainability(
            data_config=self.explainability_data_config,
            model_config=self.model_config,
            explainability_config=self.shap_config,
        )

    def deploy_model(self):
        """Deploy the best model to a SageMaker endpoint"""
        best_training_job = self.tuner.best_training_job()
        model_path = f"{self.output_path}/{best_training_job}/output/model.tar.gz"

        # Create model
        model = sagemaker.model.Model(
            model_data=model_path,
            role=self.role,
            image_uri=self.xgb_estimator.image_uri,
            name=self.endpoint_name_prefix,
            predictor_cls=sagemaker.predictor.Predictor,
        )

        # Deploy model
        self.predictor = model.deploy(
            initial_instance_count=self.predictor_instance_count,
            instance_type=self.predictor_instance_type,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer(),
        )

        return self.predictor

    def cleanup(self):
        """Clean up all created resources"""
        if hasattr(self, "predictor"):
            # Delete endpoint
            self.sagemaker_session.delete_endpoint(self.predictor.endpoint_name)
            # Delete endpoint config
            self.sagemaker_session.delete_endpoint_config(
                self.predictor._get_endpoint_config_name()
            )

        # Delete model
        try:
            self.sagemaker_session.delete_model(self.model_name)
        except Exception as e:
            print(f"Error deleting model: {str(e)}")


def main():
    """
    Main function to demonstrate the usage of HospitalXGBoostTrainer
    """
    # Replace these with your actual AWS credentials and configurations
    role_arn = "your-role-arn"
    bucket_name = "your-bucket-name"

    # Initialize the trainer
    trainer = TrainModel(role_arn, bucket_name)

    # Setup training job and hyperparameter tuning
    trainer.setup_training_job()
    trainer.setup_hyperparameter_tuning()

    # Example paths - replace with your actual S3 paths
    train_data_path = f"s3://{bucket_name}/data/train.csv"
    validation_data_path = f"s3://{bucket_name}/data/validation.csv"

    # Train model
    trainer.train_model(train_data_path, validation_data_path)

    # Analyze results
    results = trainer.analyze_results()
    print("Hyperparameter tuning results:")
    print(results)

    # Setup and run bias analysis
    train_df = pd.read_csv(train_data_path)
    train_df_cols = train_df.columns.to_list()

    trainer.setup_bias_analysis(train_data_path, train_df_cols)
    trainer.run_bias_analysis()

    # Setup and run explainability analysis
    baseline_data = [list(train_df.drop(["wait_time"], axis=1).mean())]
    trainer.setup_model_explainability(train_data_path, train_df_cols, baseline_data)
    trainer.run_explainability_analysis()

    # Deploy model
    predictor = trainer.deploy_model()
    print(f"Model deployed at endpoint: {predictor.endpoint_name}")

    # Example prediction
    sample_data = train_df.drop(["wait_time"], axis=1).iloc[0].to_list()
    prediction = predictor.predict(sample_data)
    print(f"Sample prediction: {prediction}")

    # Cleanup resources
    trainer.cleanup()


if __name__ == "__main__":
    main()
