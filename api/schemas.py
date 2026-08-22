from pydantic import BaseModel
from typing import List

class HealthResponse(BaseModel):
    status: str
    service: str
    model: str
    alias: str

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    alias: str
    task: str

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BBox

class PredictionResponse(BaseModel):
    detections: List[Detection]
