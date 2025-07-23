# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import io

app = FastAPI(title="Heart Attack Risk Prediction API")

# Загрузка лучшей модели
model = joblib.load('models/CatBoost_pipeline.joblib')

class PredictionInput(BaseModel):
    features: dict

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_single")
async def predict_single(input_data: PredictionInput):
    try:
        df = pd.DataFrame([input_data.features])
        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
