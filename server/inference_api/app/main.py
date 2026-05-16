from fastapi import FastAPI, HTTPException, File, UploadFile
from app.model_loader import load_production_model
from app.preprocess import preprocess_image
import torch
from prometheus_client import Counter, Histogram, make_asgi_app


app = FastAPI(title="FL Medical MLOps Inference API")

# Prometheus Metrics
PREDICTION_COUNT = Counter(
    "medical_predictions_total",
    "Total number of predictions",
    ["model_type", "diagnosis"]
)
PREDICTION_LATENCY = Histogram(
    "medical_prediction_duration_seconds",
    "Time taken for prediction",
    ["model_type"]
)

# Expose metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
def read_root():
    return {
        "message": "Federated Medical MLOps Inference API",
        "models": ["mlp", "cnn", "logreg"],
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/upload")
async def predict_upload(model_type: str, file: UploadFile = File(...)):
    """
    Dự đoán ảnh thông qua các model FL.
    - model_type: 'mlp', 'cnn', hoặc 'logreg'
    - file: Hình ảnh đầu vào (jpg, png...)
    """
    valid_models = ["mlp", "cnn", "logreg"]
    mt = model_type.lower()
    if mt not in valid_models:
        raise HTTPException(status_code=400, detail=f"model_type phải là 1 trong {valid_models}")
        
    # Tải động Model từ MLflow theo loại được truyền vào (A/B testing)
    model_info = load_production_model(mt)
    if "error" in model_info:
        raise HTTPException(status_code=500, detail=model_info["error"])
        
    # Đọc và tiền xử lý ảnh
    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)  # Trả về shape [1, 1, 28, 28]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi xử lý hình ảnh: {str(e)}")
        
    # Dàn phẳng tensor cho MLP và LogReg theo quy định của class
    if mt in ["mlp", "logreg"]:
        tensor = tensor.view(tensor.size(0), -1) # Thành [1, 784]
        
    # Thâm nhập mạng Neural
    net = model_info["network"]
    
    with PREDICTION_LATENCY.labels(model_type=mt).time():
        with torch.no_grad():
            output = net(tensor) # Output: Tensor([[0.823]])
            prob = float(output.item())
            
    diagnosis = "Pneumonia (Viêm phổi)" if prob > 0.5 else "Normal (Bình thường)"
    
    PREDICTION_COUNT.labels(model_type=mt, diagnosis=diagnosis).inc()
    
    return {
        "model_used": model_info["registry"],
        "model_version": model_info["version"],
        "pneumonia_probability": round(prob, 4),
        "diagnosis": diagnosis,
        "detail": "Suy luận trực tiếp bằng PyTorch và MLflow thành công!"
    }
