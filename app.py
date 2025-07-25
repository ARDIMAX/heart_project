from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import joblib
import io

app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API for predicting heart attack risk using machine learning model",
    version="1.0.0"
)

# Load the model
try:
    model = joblib.load('models/CatBoost_pipeline.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


class PredictionInput(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary containing feature names and their values"
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Prediction result (0 or 1)")


class BatchPredictionResponse(BaseModel):
    predictions: List[int] = Field(..., description="List of predictions (0 or 1)")


@app.get("/health")
async def health_check():
    """Check if the service is healthy and model is loaded."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=BatchPredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Make predictions on batch data from CSV file.

    The CSV file should contain all required features in the correct format.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validate that all required features are present
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        predictions = model.predict(df)
        return BatchPredictionResponse(predictions=predictions.tolist())

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_single", response_model=PredictionResponse)
async def predict_single(input_data: PredictionInput):
    """
    Make prediction on single instance.

    Expects a JSON object with feature names and their values.
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        df = pd.DataFrame([input_data.features])
        prediction = model.predict(df)
        return PredictionResponse(prediction=int(prediction[0]))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )

