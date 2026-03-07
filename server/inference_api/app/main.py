from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from minio import Minio
import json
app = FastAPI()

BUCKET_NAME = "fl-artifacts"

def connect_minio():
    client = Minio(
        Endpoint="minio:9001",
        access_key="admin",
        secret_key="admin12345",
        secure=False
    )
    return client

client = connect_minio()
def read_champion_json(client_minio):
    path_object = "registry/champion.json"
    response = client_minio.get_object(BUCKET_NAME, path_object)
    champion_data = response.read().decode('utf-8')
    path_model = json.loads(champion_data).get("model_path")
    return  path_model
def get_model_from_minio(client_minio):
    path_model = read_champion_json(client_minio)
    client_minio.get_object(BUCKET_NAME, path_model)


@app.post("/create_product/")
async def create_product(name: str, price: float):
    return {"name": name, "price": price}

@app.get("/get_product/{product_id}")
async def get_product(product_id: int):
    return f"Product ID: {product_id} - Name: Sample Product - Price: $9.99"

@app.post("/test_send_message/")
async def test_send_message(message):
    return {"message": message}

