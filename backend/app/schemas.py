from pydantic import BaseModel, Field
from typing import Dict


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Network traffic features"
    )


class PredictionResponse(BaseModel):
    prediction: str
    target: int
    confidence: float
    risk_level: str
    message: str