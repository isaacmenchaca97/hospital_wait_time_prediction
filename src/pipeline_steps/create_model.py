import sagemaker
from sagemaker.workflow.steps import CreateModelStep


class ModelCreateStep:
    def __init__(self, pipeline_config, train_step):
        """Initialize model creation step

        Args:
            pipeline_config: Instance of PipelineConfig with configuration
            train_step: TrainingStep object
        """
        self.config = pipeline_config
        self.train_step = train_step

    def create_model(self):
        """Create model with training image version 1.3-1"""
        # Create a SageMaker model
        self.model = sagemaker.model.Model(
            image_uri=self.config.training_image,
            model_data=self.train_step.properties.ModelArtifacts.S3ModelArtifacts,
            sagemaker_session=self.config.sess,
            role=self.config.sagemaker_role,
        )

        # Specify model deployment instance type
        self.inputs = sagemaker.inputs.CreateModelInput(
            instance_type=self.config.deploy_instance_type_param
        )

    def get_create_model_step(self):
        """Create the model creation step for the pipeline

        Return:
            CreateModelStep: The configured create model step
        """
        self.create_model()

        return CreateModelStep(name="CreateModel", model=self.model, inputs=self.inputs)
