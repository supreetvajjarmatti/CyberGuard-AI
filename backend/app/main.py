from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import (
    PredictionRequest,
    PredictionResponse
)

from backend.app.services.prediction import predictor


app = FastAPI(
    title="CyberGuard-AI API",
    description="Machine Learning Based Network Intrusion Detection System",
    version="1.0.0"
)


# Allow React frontend to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():

    return {
        "project": "CyberGuard-AI",
        "status": "Running",
        "message": "Network Intrusion Detection API is active."
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "XGBoost",
        "features": len(predictor.feature_names)
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:

        result = predictor.predict(
            request.features
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )