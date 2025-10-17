# utils/schemas.py

from pydantic import BaseModel, Field
from typing import List

class PredictionResponse(BaseModel):
    """نتيجة تنبؤ لصورة واحدة"""
    base_name: str = Field(..., description="اسم الصورة")
    class_index: int = Field(..., description="رقم الفئة المتوقعة")
    class_name: str = Field(..., description="اسم الفئة المتوقعة")
    confidence: float = Field(..., ge=0.0, le=1.0, description="نسبة الثقة في التنبؤ")

class PredictionsResponse(BaseModel):
    """نتائج متعددة (لو فيه أكثر من صورة)"""
    predictions: List[PredictionResponse]
