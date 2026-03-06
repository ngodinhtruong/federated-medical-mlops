import io
import os
import numpy as np
import torch
from minio import Minio

from model.MLP import MLP


def connect_minio_client():
    return Minio(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("ACCESS_KEY", "admin"),
        secret_key=os.getenv("SECRET_KEY", "admin12345"),
        secure=False
    )

def load_model_from_minio():
    client_minio = connect_minio_client()
    bucket_name = os.getenv("MODEL_BUCKET", "fl-artifacts")
    registry_path = os.getenv("MODEL_REGISTRY_PATH", "model_registry/")
    champion_json = os.getenv("CHAMPION_JSON", "champion.json")

    champion_obj = client_minio.get_object(bucket_name, registry_path + champion_json)

    