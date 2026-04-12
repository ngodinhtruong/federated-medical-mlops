from fastapi import FastAPI, HTTPException, File, UploadFile
from app.model_loader import load_production_model
from app.preprocess import preprocess_image
import torch

app = FastAPI(title="FL Medical MLOps Inference API")

@app.get("/")
def read_root():
    return {"message": "Welcome to Federated Medical MLOps Inference API"}

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
    
    with torch.no_grad():
        output = net(tensor) # Output: Tensor([[0.823]])
        prob = float(output.item())
        
    diagnosis = "Pneumonia (Viêm phổi)" if prob > 0.5 else "Normal (Bình thường)"
    
    return {
        "model_used": model_info["registry"],
        "model_version": model_info["version"],
        "pneumonia_probability": round(prob, 4),
        "diagnosis": diagnosis,
        "detail": "Suy luận trực tiếp bằng PyTorch và MLflow thành công!"
    }
