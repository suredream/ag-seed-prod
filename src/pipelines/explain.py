import os
import requests
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-001"

def build_prompt(feature_series, names, shap_vals, predictions: list) -> str:
    # contributions = [(feat, shap_vals[i], feature_series[feat]) for i, feat in enumerate(feature_series.index)]
    contributions = [(name, shap_v, value) for name, shap_v, value in zip(names, shap_vals, feature_series)]
    sorted_contrib = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    explanation = "\n".join([
        f"- Feature `{f}` with value `{v}` contributed {'positively' if s > 0 else 'negatively'} ({s:.3f})"
        for f, s, v in sorted_contrib
    ])

    prompt = f"""

You are a business analyst helping stakeholders understand why a machine learning model predicted a specific value for a seed product.

Given the model prediction and SHAP values, explain the result in clear, concise natural business language, suitable for non-technical readers in agriculture or seed product development.

Predictions:
- Trait-based model predicted {predictions[1]:.2f} units.
- Residual model (product-specific correction) added {predictions[2]:.2f} units.
- Final predicted sales: {predictions[0]:.2f} units.

Top trait drivers:

{explanation}

Replace technical feature names with intuitive business terms:

PROTECTION_SCORE: refers to the strength or robustness of the seed’s built-in protective traits (e.g., pest resistance, disease tolerance).

RELATIVE_MATURITY: indicates how early or late the seed variety matures, relative to standard benchmarks.

Your explanation should:

Describe how each feature likely influenced the model’s prediction.

Use business-relevant logic (e.g., "products with stronger protection tend to perform better").

Avoid technical jargon like “SHAP” or “baseline”.

Conclude with 1-liner insight on the sample’s value profile.

"""
    return prompt

def build_business_prompt(product_id, shap_summary, prediction_context):
    """
    Generate a prompt string for LLM to explain SHAP and residual-based prediction in business language.

    Parameters:
    - product_id (str): Identifier of the product
    - shap_summary (list of dict): List like [{"feature": str, "value": Any, "impact": float}]
    - prediction_context (dict): Includes keys:
        - predicted_units (float)
        - trait_model_output (float)
        - residual_model_output (float)

    Returns:
    - str: A complete prompt ready for LLM input
    """

    shap_lines = "\n".join([
        f"- {item['feature']}: value = {item['value']}, impact = {item['impact']:+.1f}"
        for item in shap_summary
    ])

    prompt = f"""
You are a business analyst assistant. Based on the SHAP analysis and model outputs below, generate a clear, business-friendly explanation for why this product ({product_id}) is predicted to perform well or poorly in the current season.

Data:
- Trait-based model predicted {prediction_context['trait_model_output']:.1f} units.
- Residual model (product-specific correction) added {prediction_context['residual_model_output']:.1f} units.
- Final predicted sales: {prediction_context[0]:.2f} units.

Top trait drivers:
{shap_lines}

Your output should be a paragraph explaining the sales expectation to a non-technical product manager.
""".strip()

    return prompt

  # You can feed this into an OpenAI API or other LLM tool
def call_explain(prompt: str) -> str:
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