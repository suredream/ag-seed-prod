from fastapi import FastAPI
from pydantic import BaseModel

from src.pipelines.explain import call_explain, build_prompt

# Initialize FastAPI app
app = FastAPI(
    title="Model Prediction API",
    description="Predict values and return confidence intervals"
)

# Define data models for request and response
class PredictRequest(BaseModel):
    """
    Request model for prediction endpoint.
    """
    data: dict

class ShapRequest(BaseModel):
    """
    Request model for SHAP explanation endpoint.
    """
    features: list[float]
    names: list[str]
    shap: list[float]
    prediction: list[float]

class PredictResponse(BaseModel):
    """
    Response model for prediction endpoint.
    """
    prediction: list[float]

# Define API endpoints
@app.post("/explain_shap")
def explain_shap(req: ShapRequest):
    """
    Generates an explanation for a single prediction using SHAP values and a GenAI model.

    Args:
        req (ShapRequest): Request containing feature values, feature names,
                           SHAP values, and the model's prediction.

    Returns:
        dict: A dictionary containing the prediction, the prompt used for
              explanation, and the generated explanation.
    """
    input_vals = req.features
    names = req.names
    shap_values = req.shap
    prediction = req.prediction

    # Build prompt for GenAI explanation
    prompt = build_prompt(input_vals, names, shap_values, prediction)
    # Call GenAI model to get explanation
    explanation = call_explain(prompt)

    return {
        "prediction": prediction,
        "prompt": prompt,
        "explanation": explanation
    }