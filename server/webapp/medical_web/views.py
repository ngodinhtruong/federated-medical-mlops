from django.conf import settings
from django.shortcuts import render

import requests


def index(request):
    context = {
        "api_url": settings.FASTAPI_URL,
        "models": [
            {"value": "cnn", "label": "CNN", "hint": "Khuyến nghị cho ảnh X-ray"},
            {"value": "mlp", "label": "MLP", "hint": "Baseline neural network"},
            {"value": "logreg", "label": "LOGREG", "hint": "Baseline tuyến tính"},
        ],
        "result": None,
        "error": None,
        "selected_model": "cnn",
    }

    if request.method == "POST":
        model_type = request.POST.get("model_type", "cnn")
        image = request.FILES.get("image")
        context["selected_model"] = model_type

        if image is None:
            context["error"] = "Vui lòng chọn một ảnh X-ray để dự đoán."
        else:
            try:
                response = requests.post(
                    f"{settings.FASTAPI_URL}/predict/upload",
                    params={"model_type": model_type},
                    files={"file": (image.name, image.read(), image.content_type)},
                    timeout=30,
                )
                if response.ok:
                    result = response.json()
                    prob = float(result.get("pneumonia_probability", 0))
                    result["probability_percent"] = round(prob * 100, 1)

                    if prob >= 0.65:
                        result["clinical_level"] = "high"
                        result["clinical_label"] = "Nguy cơ viêm phổi cao"
                        result["clinical_note"] = "Nên ưu tiên bác sĩ đọc phim và đối chiếu triệu chứng lâm sàng."
                    elif prob <= 0.45:
                        result["clinical_level"] = "low"
                        result["clinical_label"] = "Nguy cơ viêm phổi thấp"
                        result["clinical_note"] = "Kết quả ủng hộ bình thường, vẫn cần bác sĩ xác nhận nếu có triệu chứng."
                    else:
                        result["clinical_level"] = "review"
                        result["clinical_label"] = "Vùng không chắc chắn"
                        result["clinical_note"] = "Xác suất sát ngưỡng. Không nên xem đây là chẩn đoán cuối cùng."

                    context["result"] = result
                else:
                    detail = response.json().get("detail", response.text)
                    context["error"] = f"FastAPI trả về lỗi: {detail}"
            except requests.RequestException as exc:
                context["error"] = f"Không kết nối được FastAPI: {exc}"

    return render(request, "medical_web/index.html", context)
