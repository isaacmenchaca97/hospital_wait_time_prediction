import json

from sagemaker.workflow.pipeline import Pipeline

from src.pipeline_config import PipelineConfig
from src.pipeline_steps.check_evaluation import CheckEvaluationStep
from src.pipeline_steps.create_model import ModelCreateStep
from src.pipeline_steps.data_processing import DataProcessingStep
from src.pipeline_steps.deploy_model import DeployModelStep
from src.pipeline_steps.evaluate_model import EvaluateModelStep
from src.pipeline_steps.register_model import RegisterModelStep
from src.pipeline_steps.train_model import TrainModelStep

if __name__ == "__main__":
    # Initialize pipeline configuration
    pipeline_config = PipelineConfig()

    data_processing = DataProcessingStep(pipeline_config)
    process_step = data_processing.get_processing_step()

    train_model = TrainModelStep(pipeline_config, process_step)
    train_step = train_model.get_training_step()

    create_model = ModelCreateStep(pipeline_config, train_step)
    create_model_step = create_model.get_create_model_step()

    evaluate_model = EvaluateModelStep(pipeline_config, process_step, train_step)
    evaluation_step = evaluate_model.get_evaluation_step()

    register_model = RegisterModelStep(pipeline_config, train_step)
    register_step = register_model.get_register_model_step()

    deploy_model = DeployModelStep(pipeline_config, register_step)
    lambda_deploy_step = deploy_model.get_lambda_step()

    check_evaluation = CheckEvaluationStep(
        pipeline_config, evaluation_step, create_model_step, register_step, lambda_deploy_step
    )
    condition_step = check_evaluation.get_condition_step()

    # Create the Pipeline with all component steps and parameters
    pipeline = Pipeline(
        name=pipeline_config.pipeline_name,
        parameters=[
            pipeline_config.process_instance_type_param,
            pipeline_config.process_instance_count_param,
            pipeline_config.train_instance_type_param,
            pipeline_config.train_instance_count_param,
            pipeline_config.predictor_instance_type_param,
            pipeline_config.predictor_instance_count_param,
            pipeline_config.deploy_instance_type_param,
            pipeline_config.deploy_instance_count_param,
            pipeline_config.model_approval_status_param,
        ],
        steps=[process_step, train_step, evaluation_step, condition_step],
        sagemaker_session=pipeline_config.sess,
    )

    # Create a new or update existing Pipeline
    pipeline.upsert(role_arn=pipeline_config.sagemaker_role)

    # Full Pipeline description
    pipeline_definition = json.loads(pipeline.describe()["PipelineDefinition"])

    # Execute Pipeline
    # start_response = pipeline.start()
