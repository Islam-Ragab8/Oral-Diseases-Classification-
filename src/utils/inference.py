# utils/inference.py

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Union

from src.utils.config import MODEL, config
from src.utils.schemas import PredictionResponse, PredictionsResponse


def preprocess_image(image_path: str) -> torch.Tensor:
    
    transform = transforms.Compose([
        transforms.Resize(config.model.image_size),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)


def predict_images(image_paths: Union[str, List[str]]) -> PredictionsResponse:
    
    if MODEL is None:
        raise ValueError("❌ Model not loaded. Please check config.MODEL_PATH.")

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    tensors = [preprocess_image(p) for p in image_paths]
    batch = torch.stack(tensors)

    with torch.no_grad():
        outputs = MODEL(batch)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    results = []
    for path, idx, prob in zip(image_paths, preds, probs):
        base_name = os.path.basename(path)
        confidence = float(prob[idx].item())

        # تأكد إن labels موجودة وعددها كفاية
        if config.model.labels and idx.item() < len(config.model.labels):
            class_name = config.model.labels[idx]
        else:
            class_name = f"class_{idx.item()}"  # fallback لو labels ناقصة

        results.append(
            PredictionResponse(
                base_name=base_name,
                class_index=int(idx.item()),
                class_name=class_name,
                confidence=confidence
            )
        )

    return PredictionsResponse(predictions=results)
