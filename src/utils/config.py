# utils/config.py

import os
import json
import torch
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_PATH = BASE_DIR / "assets"
MODEL_PATH = ASSETS_PATH / "model.pth"
LABELS_PATH = ASSETS_PATH / "labels.json"


if MODEL_PATH.exists():
    MODEL = torch.load(MODEL_PATH, map_location="cpu")
    MODEL.eval()
else:
    MODEL = None
    print(f"⚠️ Warning: Model not found at {MODEL_PATH}")

if LABELS_PATH.exists():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = json.load(f)

# --------- إعدادات المشروع ---------
class ModelConfig(BaseModel):
    labels: List[str] = Field(default_factory=lambda: ["Healthy", "Caries", "Gingivitis", "Leukoplakia"])
    image_size: tuple = Field(default=(224, 224), description="Input image size expected by the model")
    model_path: Path = Field(default=MODEL_PATH, description="Path to the trained model")

class InferenceConfig(BaseModel):
    device: str = Field(default="cpu", description="Device to run inference on")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence to accept prediction")

class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()

config = AppConfig()
