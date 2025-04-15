from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.lambda_step import LambdaStep

from src.pipeline_config import SCRIPTS_DIR


class DeployModelStep:
    def __init__(self, pipeline_config, register_step):
        """Initialize deploy model step

        Args:
            pipeline_config: Instance of PipelineConfig with configuration
            regiaster_step: RegisterModel object
        """
        self.config = pipeline_config
        self.register_step = register_step

    def create_lambda_helper(self):
        """Define Lambda helper class"""
        self.func = Lambda(
            function_name="sagemaker-hospital-wait-time-lambda-step",
            execution_role_arn=self.config.sagemaker_role,
            script=f"{SCRIPTS_DIR}/lambda_deployer.py",
            handler="lambda_deployer.lambda_handler",
            timeout=600,
            memory_size=10240,
        )

    def get_lambda_step(self):
        """Create th lambda step for the pipeline

        Return:
            LambdaStep: the configured lambda step
        """
        self.create_lambda_helper()

        return LambdaStep(
            name="DeployModel",
            lambda_func=self.func,
            inputs={
                "model_name": self.config.pipeline_model_name,
                "endpoint_config_name": self.config.endpoint_config_name,
                "endpoint_name": self.config.endpoint_name,
                "model_package_arn": self.register_step.steps[0].properties.ModelPackageArn,
                "role": self.config.sagemaker_role,
                "instance_type": self.config.deploy_instance_type_param,
                "instance_count": self.config.deploy_instance_count_param,
            },
        )
