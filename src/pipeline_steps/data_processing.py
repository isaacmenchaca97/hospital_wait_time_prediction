from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep

from src.pipeline_config import SCRIPTS_DIR


class DataProcessingStep:
    def __init__(self, pipeline_config):
        """Initialize data processing step with pipeline setup configuration.

        Args:
            pipeline_config: Instance of HospitalWaitTimePipelineSetUp with configuration
        """
        self.config = pipeline_config

    def create_processor(self):
        """Create SKLearn processor with setup configuration."""
        self.sklearn_processor = SKLearnProcessor(
            framework_version="0.23-1",
            role=self.config.sagemaker_role,
            instance_count=self.config.process_instance_count_param,
            instance_type=self.config.process_instance_type_param,
            base_job_name=f"{self.config.base_job_name_prefix}-processing",
            dependencies=["gender-guesser", "gensim"]
        )

    def upload_preprocessing_script(self):
        """Upload preprocessing script to S3."""
        self.config.s3_client.upload_file(
            Filename=f"{SCRIPTS_DIR}/process.py",
            Bucket=self.config.write_bucket,
            Key=f"{self.config.write_prefix}/scripts/process.py",
        )

    def get_processing_step(self):
        """Create the processing step for the pipeline.

        Returns:
            ProcessingStep: The configured processing step
        """
        self.create_processor()
        self.upload_preprocessing_script()

        return ProcessingStep(
            name="DataProcessing",
            processor=self.sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=self.config.hospital_data_uri, destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    destination=f"{self.config.processing_output_uri}/train_data",
                    output_name="train_data",
                    source="/opt/ml/processing/train",
                ),
                ProcessingOutput(
                    destination=f"{self.config.processing_output_uri}/validation_data",
                    output_name="validation_data",
                    source="/opt/ml/processing/val",
                ),
                ProcessingOutput(
                    destination=f"{self.config.processing_output_uri}/test_data",
                    output_name="test_data",
                    source="/opt/ml/processing/test",
                ),
                ProcessingOutput(
                    destination=f"{self.config.processing_output_uri}/processed_data",
                    output_name="processed_data",
                    source="/opt/ml/processing/full",
                ),
            ],
            job_arguments=[
                "--train-ratio",
                "0.8",
                "--validation-ratio",
                "0.1",
                "--test-ratio",
                "0.1",
            ],
            code=f"{self.config.scripts_uri}/process.py",
        )
