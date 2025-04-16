from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet


class CheckEvaluationStep:
    def __init__(
        self,
        pipeline_config,
        evaluation_step,
        create_model_step,
        register_step,
        lambda_deploy_step,
    ):
        """Initialize check evaluation step

        Args:
            pipeline_config: Instance of PipelineConfig with configuration

        """
        self.config = pipeline_config
        self.evaluation_step = evaluation_step
        self.create_model_step = create_model_step
        self.register_step = register_step
        self.lambda_deploy_step = lambda_deploy_step

    def create_condition(self):
        """Define condition less than threshols"""
        self.cond_lte = ConditionLessThanOrEqualTo(
            left=JsonGet(
                step_name=self.evaluation_step.name,
                property_file=self.evaluation_step.property_files[0],
                json_path="regression_metrics.test.rmse",
            ),
            right=0.4,  # Threshold to compare model performance against
        )

    def get_condition_step(self):
        """Create the condition step for the pipeline

        Return:
            ConditionStep: The configured condition step
        """
        self.create_condition()

        return ConditionStep(
            name="CheckEvaluation",
            conditions=[self.cond_lte],
            if_steps=[
                self.create_model_step,
                self.register_step,
                self.lambda_deploy_step,
            ],
            else_steps=[],
        )
