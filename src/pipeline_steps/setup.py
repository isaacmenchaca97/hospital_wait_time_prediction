#!/usr/bin/env python3

import boto3
import sagemaker
from sagemaker.workflow.parameters import ParameterBoolean, ParameterInteger, ParameterString

from src.config import INSTANCE_TYPE_M4_XL


class HospitalWaitTimePipelineSetUp:
    def __init__(self):
        # Initialize SageMaker session and clients
        self.sess = sagemaker.Session()
        self.write_bucket = self.sess.default_bucket()
        self.write_prefix = "hospital_wait_time_prediction"

        self.region = self.sess.boto_region_name
        self.s3_client = boto3.client("s3", region_name=self.region)
        self.sm_client = boto3.client("sagemaker", region_name=self.region)
        self.sm_runtime_client = boto3.client("sagemaker-runtime")

        # Get SageMaker execution role
        self.sagemaker_role = sagemaker.get_execution_role()

        # Set up S3 paths
        self.setup_s3_paths()

        # Set pipeline parameters
        self.setup_pipeline_params()

        # Set instance configurations
        self.setup_instance_config()

        # Set pipeline names
        self.setup_pipeline_names()

    def setup_s3_paths(self):
        """Setup all S3 paths used in the pipeline"""
        self.read_bucket = self.sess.default_bucket()
        self.read_prefix = f"{self.write_prefix}/raw"

        # S3 locations
        self.raw_data_key = f"s3://{self.read_bucket}/{self.read_prefix}"
        self.processed_data_key = f"{self.write_prefix}/processed"
        self.train_data_key = f"{self.write_prefix}/train"
        self.validation_data_key = f"{self.write_prefix}/validation"
        self.test_data_key = f"{self.write_prefix}/test"

        # Full S3 paths
        self.raw_data_uri = f"{self.raw_data_key}/basededatos_50K.xlsx"
        self.output_data_uri = f"s3://{self.write_bucket}/{self.write_prefix}/"
        self.scripts_uri = f"s3://{self.write_bucket}/{self.write_prefix}/scripts"
        self.estimator_output_uri = f"s3://{self.write_bucket}/{self.write_prefix}/training_jobs"
        self.processing_output_uri = (
            f"s3://{self.write_bucket}/{self.write_prefix}/processing_jobs"
        )
        self.model_eval_output_uri = f"s3://{self.write_bucket}/{self.write_prefix}/model_eval"
        self.clarify_bias_config_output_uri = (
            f"s3://{self.write_bucket}/{self.write_prefix}/model_monitor/bias_config"
        )
        self.clarify_explainability_config_output_uri = (
            f"s3://{self.write_bucket}/{self.write_prefix}/model_monitor/explainability_config"
        )
        self.bias_report_output_uri = (
            f"s3://{self.write_bucket}/{self.write_prefix}/clarify_output/pipeline/bias"
        )
        self.explainability_report_output_uri = (
            f"s3://{self.write_bucket}/{self.write_prefix}/clarify_output/pipeline/explainability"
        )

    def setup_pipeline_names(self):
        """Setup names for pipeline components"""
        self.pipeline_name = "HospitalWaitTimePredictionXGBPipeline"
        self.pipeline_model_name = "hospital-wait-time-prediction-xgb-pipeline"
        self.model_package_group_name = "hospital-wait-time-prediction-xgb-model-group"
        self.base_job_name_prefix = "hospital-wait-time-prediction"
        self.endpoint_config_name = f"{self.pipeline_model_name}-endpoint-config"
        self.endpoint_name = f"{self.pipeline_model_name}-endpoint"

        # Set target column
        self.target_col = "tiempo_total"

    def setup_instance_config(self):
        """Setup instance types and counts"""
        self.process_instance_type = INSTANCE_TYPE_M4_XL
        self.train_instance_count = 1
        self.train_instance_type = INSTANCE_TYPE_M4_XL
        self.predictor_instance_count = 1
        self.predictor_instance_type = INSTANCE_TYPE_M4_XL
        self.clarify_instance_count = 1
        self.clarify_instance_type = INSTANCE_TYPE_M4_XL

    def setup_pipeline_params(self):
        """Setup pipeline parameters"""
        # Instance type parameters
        self.process_instance_type_param = ParameterString(
            name="ProcessingInstanceType",
            default_value=self.process_instance_type,
        )

        self.train_instance_type_param = ParameterString(
            name="TrainingInstanceType",
            default_value=self.train_instance_type,
        )

        self.train_instance_count_param = ParameterInteger(
            name="TrainingInstanceCount", default_value=self.train_instance_count
        )

        self.deploy_instance_type_param = ParameterString(
            name="DeployInstanceType",
            default_value=self.predictor_instance_type,
        )

        self.deploy_instance_count_param = ParameterInteger(
            name="DeployInstanceCount", default_value=self.predictor_instance_count
        )

        self.clarify_instance_type_param = ParameterString(
            name="ClarifyInstanceType",
            default_value=self.clarify_instance_type,
        )

        # Model bias check parameters
        self.skip_check_model_bias_param = ParameterBoolean(
            name="SkipModelBiasCheck", default_value=False
        )

        self.register_new_baseline_model_bias_param = ParameterBoolean(
            name="RegisterNewModelBiasBaseline", default_value=False
        )

        self.supplied_baseline_constraints_model_bias_param = ParameterString(
            name="ModelBiasSuppliedBaselineConstraints", default_value=""
        )

        # Model explainability check parameters
        self.skip_check_model_explainability_param = ParameterBoolean(
            name="SkipModelExplainabilityCheck", default_value=False
        )

        self.register_new_baseline_model_explainability_param = ParameterBoolean(
            name="RegisterNewModelExplainabilityBaseline", default_value=False
        )

        self.supplied_baseline_constraints_model_explainability_param = ParameterString(
            name="ModelExplainabilitySuppliedBaselineConstraints", default_value=""
        )

        # Model approval parameter
        self.model_approval_status_param = ParameterString(
            name="ModelApprovalStatus", default_value="Approved"
        )


if __name__ == "__main__":
    # Initialize the pipeline
    # pipeline = HospitalWaitTimePipelineSetUp()

    # Additional pipeline steps will be implemented here
    # Such as:
    # - Data Processing
    # - Train Model
    # - Evaluate Model
    # - Create Model
    # - Register Model
    # - Deploy Model
    pass
