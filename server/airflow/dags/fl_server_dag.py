from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime
import os


with DAG(
    dag_id="fl_server_dag",
    start_date=datetime(2024,1,1),
    schedule=None,
    catchup=False,
) as dag:


    # -------- build dataset cluster --------

    getdata = DockerOperator(
        task_id="get_data_clusters",

        image="fl-server:latest",
        command="python /opt/fl/data/get_dataset.py",

        docker_url="unix://var/run/docker.sock",
        network_mode="project_default",

        mounts=[
            Mount(
                source=r"D:/DHCN/2025-2026/HK2/CongNgheMoi/project/server",
                target="/opt/fl",
                type="bind",
            ),
        ],

        environment={
            "ROLE": "server",
            "DATA_SEED": os.getenv("DATA_SEED", "1"),

            "FL_SERVER_ADDRESS": os.getenv("FL_SERVER_ADDRESS"),
            "FL_ROUNDS": os.getenv("FL_ROUNDS"),
            "FL_MIN_CLIENTS": os.getenv("FL_MIN_CLIENTS"),
            "FL_MIN_FIT_CLIENTS": os.getenv("FL_MIN_FIT_CLIENTS"),
            "FL_MIN_EVAL_CLIENTS": os.getenv("FL_MIN_EVAL_CLIENTS"),

            "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT"),
            "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY"),
            "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY"),
            "MINIO_BUCKET": os.getenv("MINIO_BUCKET"),

            "SERVER_LOG_LEVEL": os.getenv("SERVER_LOG_LEVEL"),
        },


        auto_remove=True,
        mount_tmp_dir=False,
    )


    # -------- run FL server --------
    command = r"""
            sh -lc '
            set -e
            python -c "import flwr; print(\"[SERVER] flwr before:\", flwr.__version__)" || true
            pip uninstall -y flwr || true
            pip install --no-cache-dir flwr==1.7.0
            python -c "import flwr; print(\"[SERVER] flwr after :\", flwr.__version__)"
            python /opt/fl/server.py
            '
            """

    # run_fl = DockerOperator(
    #     task_id="run_fl_training",

    #     image="fl-server:latest",
    #     command=command,

    #     docker_url="unix://var/run/docker.sock",
    #     network_mode="project_default",

    #     mounts=[
    #         Mount(
    #             source=r"D:/DHCN/2025-2026/HK2/CongNgheMoi/project/server",
    #             target="/opt/fl",
    #             type="bind",
    #         ),
    #     ],

    #     environment={
    #         "ROLE": os.getenv("ROLE"),
    #         "DATA_SEED": os.getenv("DATA_SEED"),

    #         # flower
    #         "FL_SERVER_ADDRESS": os.getenv("FL_SERVER_ADDRESS"),
    #         "FL_ROUNDS": os.getenv("FL_ROUNDS"),
    #         "FL_MIN_CLIENTS": os.getenv("FL_MIN_CLIENTS"),
    #         "FL_MIN_FIT_CLIENTS": os.getenv("FL_MIN_FIT_CLIENTS"),
    #         "FL_MIN_EVAL_CLIENTS": os.getenv("FL_MIN_EVAL_CLIENTS"),

    #         # minio
    #         "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT"),
    #         "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY"),
    #         "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY"),
    #         "MINIO_BUCKET": os.getenv("MINIO_BUCKET"),

    #         # logging
    #         "SERVER_LOG_LEVEL": os.getenv("SERVER_LOG_LEVEL"),
    #     },

    #     port_bindings={"8080/tcp": 8080},

    #     auto_remove=True,
    #     mount_tmp_dir=False,
    # )


    # # -------- dependency --------

    # getdata >> run_fl
    getdata 

