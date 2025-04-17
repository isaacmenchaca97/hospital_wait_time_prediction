import json
import boto3

from sagemaker.workflow.pipeline import Pipeline

from src.pipeline_config import PipelineConfig
from src.pipeline_steps.check_evaluation import CheckEvaluationStep
from src.pipeline_steps.create_model import ModelCreateStep
from src.pipeline_steps.data_processing import DataProcessingStep
from src.pipeline_steps.deploy_model import DeployModelStep
from src.pipeline_steps.evaluate_model import EvaluateModelStep
from src.pipeline_steps.register_model import RegisterModelStep
from src.pipeline_steps.train_model import TrainModelStep


def cleanup_resources(pipeline_config: PipelineConfig):
    """
    Clean up and delete SageMaker resources associated with the pipeline.

    Args:
        pipeline_config (PipelineConfig): Pipeline configuration object containing resource names
    """
    try:
        # Initialize boto3 clients
        sagemaker_client = boto3.client('sagemaker')
        lambda_client = boto3.client('lambda')

        # Delete Lambda function
        print("Cleaning up Lambda function...")
        try:
            print(f"Deleting Lambda function: sagemaker-hospital-wait-time-lambda-step")
            lambda_client.delete_function(
                FunctionName="sagemaker-hospital-wait-time-lambda-step"
            )
        except lambda_client.exceptions.ResourceNotFoundException:
            print("Lambda function does not exist")
        except Exception as e:
            print(f"Error deleting Lambda function: {str(e)}")

        # Delete endpoint
        print("Cleaning up endpoint...")
        try:
            print(f"Deleting endpoint: {pipeline_config.endpoint_name}")
            sagemaker_client.delete_endpoint(EndpointName=pipeline_config.endpoint_name)
        except sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint" not in str(e):
                raise e
            print(f"Endpoint {pipeline_config.endpoint_name} does not exist")

        # Delete endpoint configuration
        print("Cleaning up endpoint configuration...")
        try:
            print(f"Deleting endpoint configuration: {pipeline_config.endpoint_config_name}")
            sagemaker_client.delete_endpoint_config(EndpointConfigName=pipeline_config.endpoint_config_name)
        except sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint configuration" not in str(e):
                raise e
            print(f"Endpoint configuration {pipeline_config.endpoint_config_name} does not exist")

        # Delete model package group
        print("Cleaning up model package group...")
        try:
            print(f"Deleting model package group: {pipeline_config.model_package_group_name}")
            sagemaker_client.delete_model_package_group(
                ModelPackageGroupName=pipeline_config.model_package_group_name
            )
        except sagemaker_client.exceptions.ClientError as e:
            if "does not exist" not in str(e):
                raise e
            print(f"Model package group {pipeline_config.model_package_group_name} does not exist")

        # Delete model
        print("Cleaning up model...")
        try:
            print(f"Deleting model: {pipeline_config.pipeline_model_name}")
            sagemaker_client.delete_model(ModelName=pipeline_config.pipeline_model_name)
        except sagemaker_client.exceptions.ClientError as e:
            if "Could not find model" not in str(e):
                raise e
            print(f"Model {pipeline_config.pipeline_model_name} does not exist")

        # Delete pipeline
        print("Cleaning up pipeline...")
        try:
            print(f"Deleting pipeline: {pipeline_config.pipeline_name}")
            sagemaker_client.delete_pipeline(PipelineName=pipeline_config.pipeline_name)
        except sagemaker_client.exceptions.ClientError as e:
            if "does not exist" not in str(e):
                raise e
            print(f"Pipeline {pipeline_config.pipeline_name} does not exist")

        # Clean up any training jobs
        print("Cleaning up training jobs...")
        training_jobs = sagemaker_client.list_training_jobs(
            NameContains=pipeline_config.base_job_name_prefix
        )
        for job in training_jobs['TrainingJobSummaries']:
            try:
                job_name = job['TrainingJobName']
                print(f"Stopping training job: {job_name}")
                sagemaker_client.stop_training_job(TrainingJobName=job_name)
            except sagemaker_client.exceptions.ClientError as e:
                if "No active training job" not in str(e):
                    raise e
                print(f"Training job {job_name} is not active")

        print("Resource cleanup completed successfully")

    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


if __name__ == "__main__":
    # Initialize pipeline configuration
    pipeline_config = PipelineConfig()

    # data_processing = DataProcessingStep(pipeline_config)
    # process_step = data_processing.get_processing_step()

    # train_model = TrainModelStep(pipeline_config, process_step)
    # train_step = train_model.get_training_step()

    # create_model = ModelCreateStep(pipeline_config, train_step)
    # create_model_step = create_model.get_create_model_step()

    # evaluate_model = EvaluateModelStep(pipeline_config, process_step, train_step)
    # evaluation_step = evaluate_model.get_evaluation_step()

    # register_model = RegisterModelStep(pipeline_config, train_step)
    # register_step = register_model.get_register_model_step()

    # deploy_model = DeployModelStep(pipeline_config, register_step)
    # lambda_deploy_step = deploy_model.get_lambda_step()

    # check_evaluation = CheckEvaluationStep(
    #     pipeline_config, evaluation_step, create_model_step, register_step, lambda_deploy_step
    # )
    # condition_step = check_evaluation.get_condition_step()

    # # Create the Pipeline with all component steps and parameters
    # pipeline = Pipeline(
    #     name=pipeline_config.pipeline_name,
    #     parameters=[
    #         pipeline_config.process_instance_type_param,
    #         pipeline_config.process_instance_count_param,
    #         pipeline_config.train_instance_type_param,
    #         pipeline_config.train_instance_count_param,
    #         pipeline_config.predictor_instance_type_param,
    #         pipeline_config.predictor_instance_count_param,
    #         pipeline_config.deploy_instance_type_param,
    #         pipeline_config.deploy_instance_count_param,
    #         pipeline_config.model_approval_status_param,
    #     ],
    #     steps=[process_step, train_step, evaluation_step, condition_step],
    #     sagemaker_session=pipeline_config.sess,
    # )

    # # Create a new or update existing Pipeline
    # pipeline.upsert(role_arn=pipeline_config.sagemaker_role)

    # # Full Pipeline description
    # pipeline_definition = json.loads(pipeline.describe()["PipelineDefinition"])

    # Execute Pipeline
    # start_response = pipeline.start()

    # To clean up resources, uncomment the following line:
    cleanup_resources(pipeline_config)
