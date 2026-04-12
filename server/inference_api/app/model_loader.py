import os
import torch
import mlflow
from mlflow.tracking import MlflowClient

# Import các kiến trúc gốc
from model.MLP import MLP
from model.SimpleCNN import SimpleCNN
from model.LogisticRegression import LogisticRegression

# Cache lưu trữ model để không phải tải lại (.pth) từ MinIO mỗi lần có ảnh tới
_MODEL_CACHE = {}

def get_instance(model_type: str):
    """Khởi tạo cấu trúc PyTorch trống tùy loại model"""
    if model_type == "mlp":
        return MLP(input_dim=784)
    elif model_type == "cnn":
        return SimpleCNN(input_channels=1) 
    elif model_type == "logreg":
        return LogisticRegression(input_dim=784)
    else:
        raise ValueError(f"Kiến trúc {model_type} không hợp lệ.")

def load_production_model(model_type: str):
    """
    Tải model từ MLflow, khởi tạo kiến trúc, nạp weights (.pth), trả về Neural Network sống.
    """
    client = MlflowClient()
    registry_name = f"fl_model_{model_type.lower()}"
    
    try:
        # Lấy thông tin bản Production mới nhất
        versions = client.get_latest_versions(registry_name, stages=["Production"])
        if not versions:
            raise ValueError(f"Không có model Production nào cho nhánh {registry_name}")
            
        prod_version = versions[0]
        run_id = prod_version.run_id
        current_version_num = prod_version.version
        
        cache_key = f"{registry_name}_v{current_version_num}"
        
        # Nếu model này (với đúng phiên bản này) đã có trong RAM, dùng luôn!
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
            
        # Chưa có thì tải phần mềm (Artifact) từ MLflow --> S3
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        
        pth_file = None
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith(".pth"):
                    pth_file = os.path.join(root, file)
                    break
                    
        if not pth_file:
            raise FileNotFoundError("Không tìm thấy file .pth trong artifact tải vể.")
            
        # Khởi tạo khung xương (Neural Network)
        net = get_instance(model_type.lower())
        
        # Tiêm linh hồn (Weights) vào
        device = torch.device("cpu")
        flower_state_dict = torch.load(pth_file, map_location=device)
        
        # Lấy danh sách Tensor theo chiều tăng dần của param_0, param_1... (chuẩn của Flower FL)
        nds = []
        for k in sorted(flower_state_dict.keys(), key=lambda x: int(x.split("_")[1])):
            nds.append(flower_state_dict[k].cpu())
            
        # Ánh xạ theo thứ tự các lớp của model gốc
        new_state_dict = dict(zip(net.state_dict().keys(), nds))
        
        # Nạp lại chuẩn
        net.load_state_dict(new_state_dict)
        
        # Chuyển qua chế độ Suy luận (Tắt dropout, backprop...)
        net.eval()
        
        # Đóng gói và lưu Cache
        result = {
            "registry": registry_name,
            "version": current_version_num,
            "run_id": run_id,
            "network": net,
            "status": "success"
        }
        
        _MODEL_CACHE[cache_key] = result
        return result
        
    except Exception as e:
        return {"error": str(e)}