from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import timedelta
import pendulum
import os
from dotenv import dotenv_values
local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

CLIENT_ID = os.getenv("CLIENT_ID", "A")
ENV_FILE = f"/opt/fl/env/client_{CLIENT_ID}.env"

env_vars = dotenv_values(ENV_FILE)
source_mount = os.getenv("FL_SOURCE_MOUNT", "/opt/fl")
with DAG(
    dag_id="fl_client_dag",
    start_date=pendulum.now("Asia/Ho_Chi_Minh"),
    schedule_interval=None,
    catchup=False,
) as dag:

    send_status = DockerOperator(
        task_id="send_client_status",
        image="fl-client:latest",
        command="python /opt/fl/client_status.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="project_default",
        auto_remove=True,
        mount_tmp_dir=False,
        environment=env_vars,
        mounts=[
        Mount(
            source=source_mount,
            target="/opt/fl",
            type="bind",
        ),]
    )


   

    train = DockerOperator(
        task_id="train",
        image="fl-client:latest",
        command="python /opt/fl/client.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="project_default",
        auto_remove=True,
        mount_tmp_dir=False,
        environment=env_vars,
        mounts=[
        Mount(
            source=source_mount,
            target="/opt/fl",
            type="bind",
        ),]
    )
    send_status >> train
