from sagemaker.workflow.step_collections import RegisterModel


class RegisterModelStep:
    def __init__(self, pipeline_config, train_step):
        """Initialize resgister model step

        Args:
            pipeline_config: Instance of PipelineConfig with configuration
            train_step: TrainingStep object
        """
        self.config = pipeline_config
        self.train_step = train_step

    def get_register_model_step(self):
        """Create the resgister model step for the pipeline

        Return:
            RegisterModel: The configured register model step
        """
        return RegisterModel(
            name="RegisterModel",
            estimator=self.train_step.estimator,
            # Fetching S3 location where train step saved model artifacts
            model_data=self.train_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=[self.config.predictor_instance_type],
            transform_instances=[self.config.predictor_instance_type],
            model_package_group_name=self.config.model_package_group_name,
            approval_status=self.config.model_approval_status_param,
        )
