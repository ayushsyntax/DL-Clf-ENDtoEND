import os
import time
import subprocess
import boto3

from src.common.logging import logger, setup_logging

setup_logging()

def run_command(cmd_args, shell=False):
    subprocess.run(cmd_args, shell=shell, check=True)

def deploy_to_ecs():
    """
    Automated deployment script to AWS ECS Fargate.
    """
    region = os.environ.get("AWS_REGION")
    ecr_repository = os.environ.get("ECR_REPOSITORY")
    cluster_name = os.environ.get("ECS_CLUSTER")

    if not all([region, ecr_repository, cluster_name]):
        logger.error("Missing required environment variables: AWS_REGION, ECR_REPOSITORY, ECS_CLUSTER")
        return

    service_name = "brain-tumor-inference-service"
    task_family = "brain-tumor-inference-task"

    # 1. Get AWS Account ID
    logger.info("Fetching AWS Account ID...")
    sts_client = boto3.client('sts', region_name=region)
    account_id = sts_client.get_caller_identity()["Account"]
    logger.info("Account ID retrieved", account_id=account_id)

    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{ecr_repository}"

    # 2. Create ECR repo if it doesn't exist
    ecr_client = boto3.client('ecr', region_name=region)
    try:
        ecr_client.describe_repositories(repositoryNames=[ecr_repository])
        logger.info("ECR repository already exists")
    except ecr_client.exceptions.RepositoryNotFoundException:
        logger.info("Creating ECR repository...")
        ecr_client.create_repository(repositoryName=ecr_repository)
        logger.info("ECR repository created")

    # 3. Build Docker image locally
    logger.info("Building Docker image...")
    run_command(["docker", "build", "-t", "brain-tumor-inference", "."])

    # 4. Authenticate Docker to ECR
    logger.info("Authenticating Docker to ECR...")
    login_cmd = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    run_command(login_cmd, shell=True)

    # 5. Tag and push image
    logger.info("Tagging and pushing image...")
    run_command(["docker", "tag", "brain-tumor-inference:latest", f"{ecr_uri}:latest"])
    run_command(["docker", "push", f"{ecr_uri}:latest"])

    ecs_client = boto3.client('ecs', region_name=region)

    # 6. Register a new ECS task definition
    logger.info("Registering ECS task definition...")
    execution_role_arn = f"arn:aws:iam::{account_id}:role/ecsTaskExecutionRole"

    task_def_response = ecs_client.register_task_definition(
        family=task_family,
        networkMode='awsvpc',
        requiresCompatibilities=['FARGATE'],
        cpu='1024',
        memory='2048',
        executionRoleArn=execution_role_arn,
        containerDefinitions=[
            {
                'name': 'inference-container',
                'image': f"{ecr_uri}:latest",
                'portMappings': [
                    {
                        'containerPort': 8000,
                        'hostPort': 8000,
                        'protocol': 'tcp'
                    }
                ],
                'environment': [
                    {'name': 'CUDA_VISIBLE_DEVICES', 'value': '-1'}
                ],
                'essential': True,
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': f"/ecs/{task_family}",
                        'awslogs-region': region,
                        'awslogs-stream-prefix': 'ecs',
                        'awslogs-create-group': 'true'
                    }
                }
            }
        ]
    )
    task_def_arn = task_def_response['taskDefinition']['taskDefinitionArn']
    logger.info("Task definition registered", arn=task_def_arn)

    # 7. Create ECS cluster if it doesn't exist
    try:
        clusters = ecs_client.describe_clusters(clusters=[cluster_name])
        if not clusters['clusters'] or clusters['clusters'][0]['status'] == 'INACTIVE':
            logger.info("Creating ECS cluster...")
            ecs_client.create_cluster(clusterName=cluster_name)
    except Exception as e:
        logger.info("Creating ECS cluster based on exception...", error=str(e))
        ecs_client.create_cluster(clusterName=cluster_name)

    # 8. Create or Update ECS service
    logger.info("Ensuring ECS service exists...")
    ec2_client = boto3.client('ec2', region_name=region)
    vpc_response = ec2_client.describe_vpcs(Filters=[{'Name': 'isDefault', 'Values': ['true']}])

    if not vpc_response['Vpcs']:
        logger.error("No default VPC found to deploy Fargate service.")
        return

    vpc_id = vpc_response['Vpcs'][0]['VpcId']
    subnets_response = ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    subnet_ids = [subnet['SubnetId'] for subnet in subnets_response['Subnets']]

    if not subnet_ids:
        logger.error("No subnets found in default VPC.")
        return

    network_config = {
        'awsvpcConfiguration': {
            'subnets': subnet_ids,
            'assignPublicIp': 'ENABLED'
        }
    }

    try:
        services = ecs_client.describe_services(cluster=cluster_name, services=[service_name])
        if not services['services'] or services['services'][0]['status'] == 'INACTIVE':
            logger.info("Creating ECS service...")
            ecs_client.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_def_arn,
                desiredCount=1,
                launchType='FARGATE',
                networkConfiguration=network_config
            )
        else:
            logger.info("Updating existing ECS service...")
            ecs_client.update_service(
                cluster=cluster_name,
                service=service_name,
                taskDefinition=task_def_arn,
                forceNewDeployment=True,
                networkConfiguration=network_config
            )
    except ecs_client.exceptions.ServiceNotFoundException:
        logger.info("Creating ECS service...")
        ecs_client.create_service(
            cluster=cluster_name,
            serviceName=service_name,
            taskDefinition=task_def_arn,
            desiredCount=1,
            launchType='FARGATE',
            networkConfiguration=network_config
        )

    # 9. Wait for deployment to stabilize
    logger.info("Waiting for deployment to stabilize...")
    start_time = time.time()
    timeout = 300 # 5 minutes

    while True:
        services = ecs_client.describe_services(cluster=cluster_name, services=[service_name])
        deployments = services['services'][0]['deployments']
        primary_deployment = next((d for d in deployments if d['status'] == 'PRIMARY'), None)

        if primary_deployment:
            # Check desired vs running to determine stabilization
            if primary_deployment['runningCount'] == primary_deployment['desiredCount'] and primary_deployment['desiredCount'] > 0:
                # Give it a few extra seconds to ensure health checks pass
                time.sleep(10)
                logger.info("Deployment stabilized!")
                break

        if time.time() - start_time > timeout:
            logger.error("Deployment timed out!")
            break

        logger.info("Waiting... (polling every 15s)")
        time.sleep(15)

    # 10. Print public IP
    logger.info("Fetching public IP of the running task...")
    tasks_response = ecs_client.list_tasks(cluster=cluster_name, serviceName=service_name, desiredStatus='RUNNING')

    if not tasks_response['taskArns']:
        logger.warning("No running tasks found to fetch IP.")
        return

    task_details = ecs_client.describe_tasks(cluster=cluster_name, tasks=tasks_response['taskArns'])
    eni_id = None

    for attachment in task_details['tasks'][0].get('attachments', []):
        if attachment['type'] == 'ElasticNetworkInterface':
            for kv in attachment['details']:
                if kv['name'] == 'networkInterfaceId':
                    eni_id = kv['value']
                    break

    if eni_id:
        eni_details = ec2_client.describe_network_interfaces(NetworkInterfaceIds=[eni_id])
        public_ip = eni_details['NetworkInterfaces'][0].get('Association', {}).get('PublicIp')
        if public_ip:
            logger.info("DEPLOYMENT SUCCESSFUL!")
            print(f"API is live at: http://{public_ip}:8000/docs")
            print(f"Health Endpoint: http://{public_ip}:8000/health")
            print(f"Predict Endpoint: http://{public_ip}:8000/predict")
        else:
            logger.warning("Task does not have a public IP mapped.")
    else:
        logger.warning("ENI not found for task.")

if __name__ == "__main__":
    deploy_to_ecs()
