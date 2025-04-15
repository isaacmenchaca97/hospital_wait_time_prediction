from sagemaker.workflow.pipeline import Pipeline
from src.pipeline_steps.data_processing import DataProcessingStep
from src.pipeline_steps.train_model import TrainModelStep
from src.pipeline_steps.create_model import ModelCreateStep
from src.pipeline_steps.evaluate_model import EvaluateModelStep
from src.pipeline_steps.register_model import RegisterModel
from src.pipeline_steps.deploy_model import DeployModelStep
from src.pipeline_config import PipelineConfig


if __name__ == "__main__":
    pipeline_config = PipelineConfig()
