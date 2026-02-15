from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import timedelta
import pendulum
import os

local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

with DAG(
    dag_id="fl_client_dag",
    start_date=pendulum.now("Asia/Ho_Chi_Minh"),
    schedule_interval=None,
    catchup=False,
) as dag:


    common_env = {
        "ROLE": "client",
        "CLIENT_ID": os.getenv("CLIENT_ID", "A"),

        "DATA_SEED": os.getenv("DATA_SEED", "1"),
        "DATA_N": os.getenv("DATA_N", "500"),
        "VAL_RATIO": os.getenv("VAL_RATIO", "0.2"),

        "CLIENT_HAS_DATA": os.getenv("CLIENT_HAS_DATA", "true"),
        "CLIENT_CAN_TRAIN": os.getenv("CLIENT_CAN_TRAIN", "true"),

        "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT", ""),
        "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY", ""),
        "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY", ""),
        "MINIO_BUCKET": os.getenv("MINIO_BUCKET", ""),

        "FL_SERVER_ADDRESS": os.getenv("FL_SERVER_ADDRESS", ""),

        "CLIENT_DELAY_SEC": os.getenv("CLIENT_DELAY_SEC", "0"),
        "CLIENT_DELAY_JITTER_SEC": os.getenv("CLIENT_DELAY_JITTER_SEC", "0"),

        "SERVER_LOG_LEVEL": os.getenv("SERVER_LOG_LEVEL", "INFO"),
    }


    getdata = DockerOperator(
        task_id="download_client_data",

        image="fl-client:latest",
        command="python /opt/fl/data/downloadData.py",

        docker_url="unix://var/run/docker.sock",
        network_mode="project_default",

        mounts=[
            Mount(
                source=r"D:/DHCN/2025-2026/HK2/CongNgheMoi/project/clients",
                target="/opt/fl",
                type="bind",
            ),
        ],

        environment=common_env,

        auto_remove=True,
        mount_tmp_dir=False,
    )


    send_status = DockerOperator(
        task_id="send_client_status",
        image="fl-client:latest",
        command="python /opt/fl/client_status.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="project_default",
        auto_remove=True,
        mount_tmp_dir=False,
        environment={
            "CLIENT_ID": os.getenv("CLIENT_ID"),
            "CLIENT_SEED": os.getenv("CLIENT_SEED"),
            "CLIENT_HAS_DATA": os.getenv("CLIENT_HAS_DATA"),
            "CLIENT_CAN_TRAIN": os.getenv("CLIENT_CAN_TRAIN"),
            "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT"),
            "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY"),
            "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY"),
            "MINIO_BUCKET": os.getenv("MINIO_BUCKET"),
            "FL_SERVER_ADDRESS": os.getenv("FL_SERVER_ADDRESS"),
            "CLIENT_DELAY_SEC": os.getenv("CLIENT_DELAY_SEC", "0"),
            "CLIENT_DELAY_JITTER_SEC": os.getenv("CLIENT_DELAY_JITTER_SEC", "0"),
        },
        mounts=[
            # code
            Mount(
                source="D:/DHCN/2025-2026/HK2/CongNgheMoi/project/clients",
                target="/opt/fl",
                type="bind",
            ),

            # Mount(
            #     source=f"D:\DHCN\2025-2026\HK2\CongNgheMoi\project\clients\data\cluster\cluster_client_{os.getenv('CLIENT_ID')}.npz",
            #     target=f"/opt/fl/clusters/cluster_client_{os.getenv('CLIENT_ID')}.npz",
            #     type="bind",
            # ),
        ],
    )


    # wait_server_start_fl = TimeDeltaSensor(
    #     task_id="wait_server_start_fl",
    #     delta=timedelta(minutes=5),
    #     mode="reschedule",
    # )


    train = DockerOperator(
        task_id="train",
        image="fl-client:latest",
        command="python /opt/fl/client.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="project_default",
        auto_remove=True,
        mount_tmp_dir=False,
        environment=common_env,
        mounts=[
            Mount(
                source="D:/DHCN/2025-2026/HK2/CongNgheMoi/project/clients",
                target="/opt/fl",
                type="bind",
            ),
            # Mount(
            #     source=f"D:/DHCN/2025-2026/HK2/CongNgheMoi/project/clients/clusters/cluster_client_{os.getenv('CLIENT_ID')}.npz",
            #     target=f"/opt/fl/clusters/cluster_client_{os.getenv('CLIENT_ID')}.npz",
            #     type="bind",
            # ),
        ],
    )

    # send_status >> wait_server_start_fl >> train
    getdata >> send_status >> train
