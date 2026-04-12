from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import pendulum
import os
from dotenv import dotenv_values


CLIENT_ID = os.getenv("CLIENT_ID", "A")
ENV_FILE = f"/opt/fl/env/client_{CLIENT_ID}.env"

env_vars = dotenv_values(ENV_FILE)

source_mount = os.getenv("FL_SOURCE_MOUNT", "/opt/fl")

with DAG(
    dag_id="fl_client_dag",
    start_date=datetime(2024, 1, 1),
    schedule="*/5 * * * *",   
    catchup=False,
) as dag:

    ingest_data = DockerOperator(
        task_id="ingest_new_data",
        image="fl-client:latest",
        command="python /opt/fl/data/stream_data.py",
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
            ),
        ],
    )

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
            ),
        ],
    )

    check_data_quality = DockerOperator(
        task_id="check_data_quality",
        image="fl-client:latest",
        command="python /opt/fl/data/check_data_quality.py",
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
            ),
        ],
    )

    def create_train_task(model_name, server_address):
        return DockerOperator(
            task_id=f"train_{model_name.lower()}",
            image="fl-client:latest",
            command="python /opt/fl/client.py",
            docker_url="unix://var/run/docker.sock",
            network_mode="project_default",
            auto_remove=True,
            mount_tmp_dir=False,
            environment={
                **env_vars,
                "MODEL_NAME": model_name,
                "FL_SERVER_ADDRESS": server_address
            },
            mounts=[
                Mount(
                    source=source_mount,
                    target="/opt/fl",
                    type="bind",
                ),
            ],
        )

    train_mlp = create_train_task("MLP", "fl-server-mlp:8080")
    train_cnn = create_train_task("CNN", "fl-server-cnn:8080")
    train_logreg = create_train_task("LogisticRegression", "fl-server-logreg:8080")

    ingest_data >> check_data_quality >> send_status >> [train_mlp, train_cnn, train_logreg]