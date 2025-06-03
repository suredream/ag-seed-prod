import sys
import os
import toml
import requests

# Dynamically add project root (containing 'src') to PYTHONPATH
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import shap
from src.utils import load_artifacts, calculate_confidence_intervals
from src.pipelines.xgb import train_and_save, inference


# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # 你需要 export 这个变量
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-001"


app = FastAPI(title="Model Prediction API", description="Predict values and return confidence intervals")

# from fastapi.exceptions import RequestValidationError
# from fastapi.responses import JSONResponse
# from fastapi import Request

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     return JSONResponse(
#         status_code=422,
#         content={"detail": exc.errors(), "body": exc.body}
#     )


# Load model once on startup
config = toml.load("config/xgb.toml")
model, scaler, pca, X_train, X_test, y_train, y_test = load_artifacts(config)
# print(X_train.columns)

explainer = shap.Explainer(model, X_train.astype(float))

feature_cols = ['SALESYEAR', 'RELEASE_YEAR',"DISEASE_RESISTANCE", "INSECT_RESISTANCE", "PROTECTION",
                 'DROUGHT_TOLERANCE', 'BRITTLE_STALK', 'PLANT_HEIGHT', 'RELATIVE_MATURITY', 'STATE', 'LIFECYCLE']
class PredictRequest(BaseModel):
    data: dict

class ShapRequest(BaseModel):
    features: list[float]
    names: list[str]
    shap: list[float]
    prediction: float

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

def build_prompt(feature_series: pd.Series, shap_vals, prediction: float) -> str:
    contributions = [(feat, shap_vals[i], feature_series[feat]) for i, feat in enumerate(feature_series.index)]
    sorted_contrib = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    explanation = "\n".join([
        f"- Feature `{f}` with value `{v}` contributed {'positively' if s > 0 else 'negatively'} ({s:.3f})"
        for f, s, v in sorted_contrib
    ])

    prompt = f"""
A machine learning seep production forecast predicted a value of **{prediction:.2f}** for a sample.

You are a business analyst helping stakeholders understand why a machine learning model predicted a specific value for a seed product.

Given the model prediction and SHAP values, explain the result in clear, natural business language, suitable for non-technical readers in agriculture or seed product development.

{explanation}

Replace technical feature names with intuitive business terms:

PROTECTION_SCORE: refers to the strength or robustness of the seed’s built-in protective traits (e.g., pest resistance, disease tolerance).

RELATIVE_MATURITY: indicates how early or late the seed variety matures, relative to standard benchmarks.

Your explanation should:

Describe how each feature likely influenced the model’s prediction.

Use business-relevant logic (e.g., "products with stronger protection tend to perform better").

Avoid technical jargon like “SHAP” or “baseline”.

Conclude with a summary insight on the sample’s value profile.

"""
    return prompt

def call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content'].strip()

# === 主接口 ===

def build_prompt2(feature_series, names, shap_vals, prediction: float) -> str:
    # contributions = [(feat, shap_vals[i], feature_series[feat]) for i, feat in enumerate(feature_series.index)]
    contributions = [(name, shap_v, value) for name, shap_v, value in zip(names, shap_vals, feature_series)]
    sorted_contrib = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    explanation = "\n".join([
        f"- Feature `{f}` with value `{v}` contributed {'positively' if s > 0 else 'negatively'} ({s:.3f})"
        for f, s, v in sorted_contrib
    ])

    prompt = f"""
A machine learning seep production forecast predicted a value of **{prediction:.2f}** for a sample.

You are a business analyst helping stakeholders understand why a machine learning model predicted a specific value for a seed product.

Given the model prediction and SHAP values, explain the result in clear, natural business language, suitable for non-technical readers in agriculture or seed product development.

{explanation}

Replace technical feature names with intuitive business terms:

PROTECTION_SCORE: refers to the strength or robustness of the seed’s built-in protective traits (e.g., pest resistance, disease tolerance).

RELATIVE_MATURITY: indicates how early or late the seed variety matures, relative to standard benchmarks.

Your explanation should:

Describe how each feature likely influenced the model’s prediction.

Use business-relevant logic (e.g., "products with stronger protection tend to perform better").

Avoid technical jargon like “SHAP” or “baseline”.

Conclude with a summary insight on the sample’s value profile.

"""
    return prompt



@app.post("/explain_shap")
def explain_shap(req: ShapRequest):
    input_vals = req.features
    names = req.names
    shap_values = req.shap
    prediction = req.prediction

    # 转换为 DataFrame
    # input_df = pd.DataFrame([input_dict])
    # input_df = input_df[feature_cols].fillna(0).infer_objects(copy=False)

    # 模型预测
    # prediction = model.predict(input_df)[0]
    # prediction, X = inference(input_df, config)

    # 计算 SHAP 值
    # shap_vals = explainer(X)
    # shap_vals = shap_vals.values[0].tolist()
    # prediction = float(prediction[0])

#     # 构建 prompt & 调用 Gemini
    # new_df = pd.DataFrame(X, columns=config['features']['final'])
    print(input_vals)
    print(shap_values)
    print(prediction)
    prompt = build_prompt2(input_vals,names, shap_values, prediction)
    print('prompt')
    print(prompt)
    explanation = call_openrouter(prompt)

    return {
        "prediction": prediction,
        # "top_features": sorted(
        #     [(f, float(v)) for f, v in zip(input_df.columns, shap_values)],
        #     key=lambda x: abs(x[1]),
        #     reverse=True
        # )[:5],
        "prompt": prompt,
        "explanation": explanation
    }


@app.post("/explain")
def explain_prediction(req: PredictRequest):
    input_dict = req.data

    # 转换为 DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_cols].fillna(0).infer_objects(copy=False)

    # 模型预测
    # prediction = model.predict(input_df)[0]
    prediction, X = inference(input_df, config)

    # 计算 SHAP 值
    shap_vals = explainer(X)
    shap_vals = shap_vals.values[0].tolist()
    prediction = float(prediction[0])

#     # 构建 prompt & 调用 Gemini
    new_df = pd.DataFrame(X, columns=config['features']['final'])
    prompt = build_prompt(new_df.iloc[0], shap_vals, prediction)
    explanation = call_openrouter(prompt)

    return {
        "prediction": prediction,
        # "top_features": sorted(
        #     [(f, float(v)) for f, v in zip(input_df.columns, shap_values)],
        #     key=lambda x: abs(x[1]),
        #     reverse=True
        # )[:5],
        "prompt": prompt,
        "explanation": explanation
    }