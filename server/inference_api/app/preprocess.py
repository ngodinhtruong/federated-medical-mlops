import io
import numpy as np
import torch
from PIL import Image

IMG_SIZE = 28


def preprocess_image(image_bytes):

    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    img = img.resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img).astype("float32") / 255.0

    arr = arr.reshape(1, 1, IMG_SIZE, IMG_SIZE)

    return torch.tensor(arr)