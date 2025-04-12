from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.xgboost.estimator import XGBoost
from src.pipeline_config import SCRIPTS_DIR


class TrainModelStep:
    def __init__(self, pipeline_config, process_step):
        """Initialize the XGBoost trainer with with pipeline setup configuration.

        Args:
            pipeline_config: Instance of PipelineConfig with configuration
            process_step: ProcessingStep object
        """
        self.config = pipeline_config
        self.process_step = process_step

    def create_estimator(self):
        """Create XGBoost training job with default hyperparameters"""
        self.static_hyperparameters = {
            "eval_metric": "rmse",
            "objective": "reg:squarederror",
            "num_round": "100",
            "max_depth": "6",
            "subsample": "0.9",
            "colsample_bytree": "0.8",
            "eta": "0.3",
        }

        self.xgb_estimator = XGBoost(
            entry_point=f"{SCRIPTS_DIR}/train.py",
            output_path=self.config.estimator_output_uri,
            code_location=self.config.estimator_output_uri,
            hyperparameters=self.static_hyperparameters,
            role=self.config.sagemaker_role,
            # Fetch instance type and count from pipeline parameters
            instance_count=self.config.train_instance_count,
            instance_type=self.config.train_instance_type,
            framework_version="1.3-1",
        )

    def get_training_step(self):
        """Create the training step for the pipeline

        Return:
            TrainingStep: The configured train step
        """
        self.create_estimator()

        return TrainingStep(
            name="TrainModel",
            estimator=self.xgb_estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=self.process_step.properties.ProcessingOutputConfig.Outputs[
                        "train_data"
                    ].S3Output.S3Uri,
                    content_type="csv",
                    s3_data_type="S3Prefix",
                ),
                "validation": TrainingInput(
                    s3_data=self.process_step.properties.ProcessingOutputConfig.Outputs[
                        "validation_data"
                    ].S3Output.S3Uri,
                    content_type="csv",
                    s3_data_type="S3Prefix",
                ),
            },
        )
