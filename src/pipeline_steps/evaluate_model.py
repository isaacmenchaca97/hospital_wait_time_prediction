from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep

from src.pipeline_config import SCRIPTS_DIR


class EvaluateModelStep:
    def __init__(self, pipeline_config, process_step, train_step):
        """Initialize evaluation model step

        Args:
            pipeline_config: Instance of PipelineConfig with configuration
            process_step: ProcessingStep object
            train_step: TrainingStep object
        """
        self.config = pipeline_config
        self.process_step = process_step
        self.train_step = train_step

    def create_processor(self):
        """Create ScriptProcessor with setup configuration"""
        self.eval_processor = ScriptProcessor(
            image_uri=self.config.training_image,
            command=["python3"],
            instance_type=self.config.predictor_instance_type,
            instance_count=self.config.predictor_instance_count,
            base_job_name=f"{self.config.base_job_name_prefix}-model-eval",
            sagemaker_session=self.config.sess,
            role=self.config.sagemaker_role,
        )

    def upload_evaluate_script(self):
        """Upload model evaluation script to S3"""
        self.config.s3_client.upload_fiel(
            Filename=f"{SCRIPTS_DIR}/evaluate.py",
            Bucket=self.config.write_bucket,
            Key=f"{self.config.write_prefix}/scripts/evaluate.py",
        )

    def create_property_file(self):
        """Create a property file for the evaluation results"""
        self.evaluation_report = PropertyFile(
            name="HospitalWaitTimePredReport",
            output_name="evaluation",
            path="evaluation.json",
        )

    def get_evaluation_step(self):
        """Create the evaluation step for the pipeline

        Return:
            ProcessingStep: The configured evaluate model step
        """
        self.create_processor()
        self.upload_evaluate_script()
        self.create_property_file()

        return ProcessingStep(
            name="EvaluateModel",
            processor=self.eval_processor,
            inputs=[
                ProcessingInput(
                    # Fetch S3 location where train step saved model artifacts
                    source=self.train_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    # Fetch S3 location where processing step saved test data
                    source=self.process_step.properties.ProcessingOutputConfig.Outputs[
                        "test_data"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    destination=f"{self.config.model_eval_output_uri}",
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                ),
            ],
            code=f"s3://{self.config.write_bucket}/{self.config.write_prefix}/scripts/evaluate.py",
            property_files=[self.evaluation_report],
        )
