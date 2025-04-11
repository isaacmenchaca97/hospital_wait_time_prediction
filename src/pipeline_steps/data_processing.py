#!/usr/bin/env python3

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep


class DataProcessingStep:
    def __init__(self, pipeline_setup):
        """Initialize data processing step with pipeline setup configuration.

        Args:
            pipeline_setup: Instance of HospitalWaitTimePipelineSetUp with configuration
        """
        self.setup = pipeline_setup

    def create_processor(self):
        """Create SKLearn processor with setup configuration."""
        self.sklearn_processor = SKLearnProcessor(
            framework_version="0.23-1",
            role=self.setup.sagemaker_role,
            instance_count=1,
            instance_type=self.setup.process_instance_type,
            base_job_name=f"{self.setup.base_job_name_prefix}-processing",
        )

    def upload_preprocessing_script(self):
        """Upload preprocessing script to S3.

        Args:
            script_path: Local path to the preprocessing script. Defaults to 'process.py'
        """
        self.setup.s3_client.upload_file(
            Filename="scripts/process.py",
            Bucket=self.setup.write_bucket,
            Key=f"{self.setup.write_prefix}/scripts/process.py",
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
                    source=self.setup.raw_data_uri, destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    destination=f"{self.setup.processing_output_uri}/train_data",
                    output_name="train_data",
                    source="/opt/ml/processing/train",
                ),
                ProcessingOutput(
                    destination=f"{self.setup.processing_output_uri}/validation_data",
                    output_name="validation_data",
                    source="/opt/ml/processing/val",
                ),
                ProcessingOutput(
                    destination=f"{self.setup.processing_output_uri}/test_data",
                    output_name="test_data",
                    source="/opt/ml/processing/test",
                ),
                ProcessingOutput(
                    destination=f"{self.setup.processing_output_uri}/processed_data",
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
            code=f"s3://{self.setup.write_bucket}/{self.setup.write_prefix}/scripts/process.py",
        )
