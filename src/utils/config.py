# src/utils/config.py
import json
import torch
import torch.nn as nn
from torchvision.models import resnet18
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_PATH = BASE_DIR / "assets"
MODEL_PATH = ASSETS_PATH / "models.pth"
LABELS_PATH = ASSETS_PATH / "labels.json"



class MyModel(nn.Module):
    def __init__(self, num_classes=6):
        super(MyModel, self).__init__()
        # استخدم نفس بنية الموديل اللي اتدرّب عليها
        self.model = resnet18(weights=True)  # خليه None لأن الموديل متدرّب فعلاً
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)



if LABELS_PATH.exists():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = json.load(f)
else:
    LABELS = ["Calculus", "Caries", "Gingivitis", "Ulcers", "Tooth Discoloration", "Hypodontia"]
    print(f"⚠️ Warning: labels.json not found at {LABELS_PATH}, using default labels.")


# -----------------------------
# 4️⃣ تحميل الموديل المدرب
# -----------------------------
if MODEL_PATH.exists():
    try:
        # أنشئ نسخة فاضية من الموديل نفس البنية
        model_instance = MyModel(num_classes=len(LABELS))

        # حمّل أوزان الموديل داخلها
        model_instance.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model_instance.eval()
        MODEL = model_instance
        print("✅ Model loaded successfully.")
    except Exception as e:
        import traceback
        print("❌ Error loading model:")
        traceback.print_exc()
        MODEL = None
else:
    MODEL = None
    print(f"⚠️ Warning: Model not found at {MODEL_PATH}")

class ModelConfig(BaseModel):
    labels: List[str] = Field(default_factory=lambda: LABELS)
    image_size: tuple = Field(default=(224, 224))
    model_path: Path = Field(default=MODEL_PATH)

class InferenceConfig(BaseModel):
    device: str = Field(default="cpu")
    confidence_threshold: float = Field(default=0.5)

class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()

config = AppConfig()
