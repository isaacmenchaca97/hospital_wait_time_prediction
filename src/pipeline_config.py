from pathlib import Path

import boto3
from dotenv import load_dotenv
from loguru import logger
import sagemaker
from sagemaker.image_uris import retrieve
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
SCRIPTS_DIR = PROJ_ROOT / "scripts"

INSTANCE_TYPE_M4_XL = "ml.m4.xlarge"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


class PipelineConfig:
    def __init__(self):
        # Instantiate AWS services session and client objects
        self.sess = sagemaker.Session()
        self.write_bucket = self.sess.default_bucket()
        self.write_prefix = "hospital_wait_time_prediction"

        self.read_bucket = self.sess.default_bucket()
        self.read_prefix = "hospital_wait_time_prediction/raw"

        self.region = self.sess.boto_region_name
        self.s3_client = boto3.client("s3", region_name=self.region)
        self.sm_client = boto3.client("sagemaker", region_name=self.region)
        self.sm_runtiome_client = boto3.client("sagemaker-runtime")

        # Fetch SageMaker execution role
        self.sagemaker_role = sagemaker.get_execution_role()

        # Retrieve training image
        self.training_image = retrieve(framework="xgboost", region=self.region, version="1.3-1")

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
        self.hospital_data_uri = (
            f"s3://{self.read_bucket}/{self.read_prefix}/basesdedatos_50K.xlsx"
        )
        self.output_data_uri = f"s3://{self.write_bucket}/{self.write_prefix}/"
        self.scripts_uri = f"s3://{self.write_bucket}/{self.write_prefix}/scripts"
        self.estimator_output_uri = f"s3://{self.write_bucket}/{self.write_prefix}/training_jobs"
        self.processing_output_uri = (
            f"s3://{self.write_bucket}/{self.write_prefix}/processing_jobs"
        )
        self.model_eval_output_uri = f"s3://{self.write_bucket}/{self.write_prefix}/model_eval"

    def setup_pipeline_names(self):
        """Setup names for pipeline components"""
        self.pipeline_name = "HospitalWaitTimePredPipeline"
        self.pipeline_model_name = "hospital-wait-time-pred-pipeline"
        self.model_package_group_name = "hospital-wait-time-pred-model-group"
        self.base_job_name_prefix = "hospital-pred"
        self.endpoint_config_name = f"{self.pipeline_model_name}-endpoint-config"
        self.endpoint_name = f"{self.pipeline_model_name}-endpoint"

    def setup_instance_config(self):
        """Setup instance types and counts"""
        self.process_instance_type = INSTANCE_TYPE_M4_XL
        self.train_instance_count = 1
        self.train_instance_type = INSTANCE_TYPE_M4_XL
        self.predictor_instance_count = 1
        self.predictor_instance_type = INSTANCE_TYPE_M4_XL

    def setup_pipeline_params(self):
        """Setup pipeline parameters"""
        # Set processing instance type
        self.process_instance_type_param = ParameterString(
            name="ProcessingInstanceType",
            default_value=self.process_instance_type,
        )

        # Set training instance type
        self.train_instance_type_param = ParameterString(
            name="TrainingInstanceType",
            default_value=self.train_instance_type,
        )

        # Set training instance count
        self.train_instance_count_param = ParameterInteger(
            name="TrainingInstanceCount", default_value=self.train_instance_count
        )

        # Set deployment instance type
        self.deploy_instance_type_param = ParameterString(
            name="DeployInstanceType",
            default_value=self.predictor_instance_type,
        )

        # Set deployment instance count
        self.deploy_instance_count_param = ParameterInteger(
            name="DeployInstanceCount", default_value=self.predictor_instance_count
        )

        # Set model approval param
        self.model_approval_status_param = ParameterString(
            name="ModelApprovalStatus", default_value="Approved"
        )
