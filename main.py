import sys
import os
import toml

# Dynamically add project root (containing 'src') to PYTHONPATH
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.utils import load_artifacts, calculate_confidence_intervals
from src.pipelines.xgb import train_and_save, inference

app = FastAPI(title="Model Prediction API", description="Predict values and return confidence intervals")

# Load model once on startup
config = toml.load("config/xgb.toml")
model, scaler, pca, X_train, X_test, y_train, y_test = load_artifacts(config)

feature_cols = ['SALESYEAR', 'RELEASE_YEAR',"DISEASE_RESISTANCE", "INSECT_RESISTANCE", "PROTECTION",
                 'DROUGHT_TOLERANCE', 'BRITTLE_STALK', 'PLANT_HEIGHT', 'RELATIVE_MATURITY', 'STATE', 'LIFECYCLE']
class PredictRequest(BaseModel):
    data: dict
    confidence: float = 0.95

class PredictResponse(BaseModel):
    prediction: list[float]
    # lower_bound: list[float]
    # upper_bound: list[float]

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X_input = pd.DataFrame([req.data])#, columns=feature_cols)
    preds, X = inference(X_input, config)
    # lower, upper = calculate_confidence_intervals(model, X, confidence=req.confidence)
    return {
        "prediction": preds.tolist(),
    }
