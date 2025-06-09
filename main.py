from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.utils import load_artifacts, calculate_confidence_intervals
from src.pipelines.xgb import train_and_save, inference
from src.pipelines.explain import call_explain, build_prompt


app = FastAPI(title="Model Prediction API", description="Predict values and return confidence intervals")

class PredictRequest(BaseModel):
    data: dict

class ShapRequest(BaseModel):
    features: list[float]
    names: list[str]
    shap: list[float]
    prediction: list[float]

class PredictResponse(BaseModel):
    prediction: list[float]
    # lower_bound: list[float]
    # upper_bound: list[float]

# @app.post("/predict", response_model=PredictResponse)
# def predict(req: PredictRequest):
#     X_input = pd.DataFrame([req.data])#, columns=feature_cols)
#     preds, X = inference(X_input, config)
#     # lower, upper = calculate_confidence_intervals(model, X, confidence=req.confidence)
#     return {
#         "prediction": preds.tolist(),
#     }

@app.post("/explain_shap")
def explain_shap(req: ShapRequest):
    # return {"hello": "world"}
    input_vals = req.features
    names = req.names
    shap_values = req.shap
    prediction = req.prediction

    prompt = build_prompt(input_vals, names, shap_values, prediction)
    explanation = call_explain(prompt)

    return {
        "prediction": prediction,
        "prompt": prompt,
        "explanation": explanation
    }