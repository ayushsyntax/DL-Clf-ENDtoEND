import boto3

from src.common.logging import logger, setup_logging

setup_logging()


def trigger_ecs_fargate_deployment():
    """
    Sends an update signal to the AWS ECS cluster to cycle the task.

    This ensures that the latest container image pushed to ECR is deployed
    and that the inference service is refreshed with the new model.
    """
    cluster_id = "brain-tumor-cluster"
    service_id = "brain-tumor-inference-service"

    logger.info(
        "Initiating ECS Fargate rolling deployment",
        cluster=cluster_id,
        service=service_id
    )

    try:
        ecs_client = boto3.client('ecs')
        deployment_response = ecs_client.update_service(
            cluster=cluster_id,
            service=service_id,
            force_new_deployment=True
        )

        active_deployment = deployment_response['service']['deployments'][0]['id']
        logger.info(
            "Deployment successfully registered with ECS",
            deployment_id=active_deployment
        )

    except Exception as ecs_error:
        logger.error("Deployment signal failed", error=str(ecs_error))


if __name__ == "__main__":
    trigger_ecs_fargate_deployment()
