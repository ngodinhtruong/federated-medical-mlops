import sys
sys.path.insert(0, "/opt/fl")

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException

from server_mlflow import log_cycle_to_mlflow
from docker.types import Mount
from datetime import datetime, timedelta
import os
import io
import json
from minio import Minio

source_mount = os.getenv("FL_SOURCE_MOUNT", "/opt/fl")

def _minio_client():
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "admin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "admin12345")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _find_next_cycle(**context):
    client = _minio_client()
    bucket = os.getenv("MINIO_BUCKET", "fl-artifacts")
    prefixes_env = os.getenv("MINIO_PREFIXES", "training/mlp,training/cnn,training/logreg")
    prefixes = [p.strip() for p in prefixes_env.split(",") if p.strip()]

    done_suffix = "/DONE.json"
    processed_suffix = "/PROCESSED.json"

    pending = []

    for prefix in prefixes:
        prefix_key = prefix.rstrip("/") + "/"
        done_cycles = set()
        processed_cycles = set()

        for obj in client.list_objects(bucket, prefix=prefix_key, recursive=True):
            name = obj.object_name
            if name.endswith(done_suffix):
                cycle_path = name[: -len(done_suffix)]
                done_cycles.add(cycle_path)
            elif name.endswith(processed_suffix):
                cycle_path = name[: -len(processed_suffix)]
                processed_cycles.add(cycle_path)

        local_pending = sorted(list(done_cycles - processed_cycles))
        pending.extend([(prefix, p) for p in local_pending])

    if not pending:
        raise AirflowSkipException("No new cycle found")

    selected_prefix, cycle_path = pending[0]
    cycle_id = cycle_path.split("/")[-1].replace("cycle_", "", 1)

    context["ti"].xcom_push(key="minio_prefix", value=selected_prefix)
    context["ti"].xcom_push(key="cycle_path", value=cycle_path)
    context["ti"].xcom_push(key="cycle_id", value=cycle_id)
    return cycle_id

def _log_mlflow_task(**context):
    ti = context["ti"]
    cycle_path = ti.xcom_pull(key="cycle_path", task_ids="detect_new_cycle")
    cycle_id = ti.xcom_pull(key="cycle_id", task_ids="detect_new_cycle")

    if not cycle_path or not cycle_id:
        raise AirflowSkipException("Missing cycle info")

    log_cycle_to_mlflow(cycle_path, cycle_id)
def _mark_processed(**context):
    ti = context["ti"]
    cycle_path = ti.xcom_pull(key="cycle_path", task_ids="detect_new_cycle")
    cycle_id = ti.xcom_pull(key="cycle_id", task_ids="detect_new_cycle")

    if not cycle_path or not cycle_id:
        raise AirflowSkipException("Missing cycle info")

    client = _minio_client()
    bucket = os.getenv("MINIO_BUCKET", "fl-artifacts")

    payload = {
        "cycle_id": str(cycle_id),
        "status": "processed",
        "processed_at": datetime.utcnow().isoformat() + "Z",
    }

    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    bio = io.BytesIO(raw)
    obj_name = f"{cycle_path}/PROCESSED.json"

    client.put_object(
        bucket_name=bucket,
        object_name=obj_name,
        data=bio,
        length=len(raw),
        content_type="application/json",
    )


with DAG(
    dag_id="fl_cycle_postprocess",
    start_date=datetime(2024, 1, 1),
    schedule="*/1 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
) as dag:

    detect_new_cycle = PythonOperator(
        task_id="detect_new_cycle",
        python_callable=_find_next_cycle,
    )

    run_eval = DockerOperator(
        task_id="evaluate_and_choose",
        image="fl-server:latest",
        command=(
            "sh -lc '"
            "python /opt/fl/server_evaluate.py "
            "--cycle_id \"{{ ti.xcom_pull(task_ids=\"detect_new_cycle\", key=\"cycle_id\") }}\" "
            "--bucket \"${MINIO_BUCKET}\" "
            "--prefix \"${MINIO_PREFIX}\" "
            "'"
        ),
        docker_url="unix://var/run/docker.sock",
        network_mode="project_default",
        auto_remove=True,
        mount_tmp_dir=False,
        environment={
            "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT", "minio:9000"),
            "MINIO_ACCESS_KEY": os.getenv("MINIO_ACCESS_KEY", "admin"),
            "MINIO_SECRET_KEY": os.getenv("MINIO_SECRET_KEY", "admin12345"),
            "MINIO_BUCKET": os.getenv("MINIO_BUCKET", "fl-artifacts"),
            "MINIO_PREFIX": '{{ ti.xcom_pull(task_ids="detect_new_cycle", key="minio_prefix") }}',
            "MINIO_SECURE": os.getenv("MINIO_SECURE", "false"),
        },
        mounts=[
            Mount(
                source=source_mount,
                target="/opt/fl",
                type="bind",
            ),
        ],
    )
    log_to_mlflow = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=_log_mlflow_task,
    )
    mark_processed = PythonOperator(
        task_id="mark_processed",
        python_callable=_mark_processed,
    )

    detect_new_cycle >> run_eval >> log_to_mlflow >> mark_processed