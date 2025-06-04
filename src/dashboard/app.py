import sys
import os

# Dynamically add project root (containing 'src') to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import streamlit as st
import pandas as pd
import numpy as np
import json
import toml
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.utils import load_artifacts, calculate_confidence_intervals
from scipy.stats import gaussian_kde
from xgboost import XGBRegressor
import requests

st.set_page_config(page_title="Model Analysis Dashboard", layout="wide")

config = toml.load("config/xgb.toml")
model, scaler, pca, X_train, X_test, y_train, y_test = load_artifacts(config)
# print(X_test.columns)

# -------------------------------
# Group filter selectors
# -------------------------------
st.sidebar.header("🔎 Filter by Group")

# Ensure required columns exist
required_columns = ['STATE', 'LIFECYCLE']
for col in required_columns:
    if col not in X_test.columns:
        st.error(f"Missing required column: {col} in X_test")
        st.stop()

# Group filters
state_options = sorted(X_test['STATE'].unique())
lifecycle_options = sorted(X_test['LIFECYCLE'].unique())
product_options = sorted(X_test['PRODUCT'].unique())

selected_states = st.sidebar.multiselect("Select STATE(s)", state_options, default=state_options)
selected_lifecycles = st.sidebar.multiselect("Select LIFECYCLE(s)", lifecycle_options, default=lifecycle_options)
selected_products = st.sidebar.multiselect("Select PRODUCT(s)", product_options, default=product_options)

# Filtered data
mask = (
    X_test['PRODUCT'].isin(selected_products) &
    X_test['STATE'].isin(selected_states) &
    X_test['LIFECYCLE'].isin(selected_lifecycles)
)
X_test_filtered = X_test[mask]
y_test_filtered = y_test[mask]

# Stop if no data
if len(X_test_filtered) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# Predict
X_test = X_test_filtered[config['features']['final']]
y_pred_test_filtered = model.predict(X_test)
y_pred_train = model.predict(X_train)

# -------------------------------
# Evaluation metrics
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Evaluation Metrics")
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_test_filtered))
    test_r2 = r2_score(y_test_filtered, y_pred_test_filtered)
    test_mae = mean_absolute_error(y_test_filtered, y_pred_test_filtered)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'R²', 'MAE'],
        'Train': [f'{train_rmse:.3f}', f'{train_r2:.3f}', f'{train_mae:.3f}'],
        'Test': [f'{test_rmse:.3f}', f'{test_r2:.3f}', f'{test_mae:.3f}']
    })
    st.dataframe(metrics_df, use_container_width=True)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_test_filtered, y=y_pred_test_filtered, mode='markers', name='Test'))
    min_val = min(y_test_filtered.min(), y_pred_test_filtered.min())
    max_val = max(y_test_filtered.max(), y_pred_test_filtered.max())
    fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect'))
    fig_scatter.update_layout(title='Predicted vs Actual', xaxis_title='Actual', yaxis_title='Predicted')
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("🎯 Feature Importance")
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    fig_importance = px.bar(importance_df, x='importance', y='feature', orientation='h')
    fig_importance.update_layout(title='Feature Importance', height=400)
    st.plotly_chart(fig_importance, use_container_width=True)

# -------------------------------
# Residual plot
# -------------------------------
st.subheader("📉 Residual Distribution")

residuals = y_test_filtered - y_pred_test_filtered
hist_data = np.histogram(residuals, bins=30)
x_hist = hist_data[1]
y_hist = hist_data[0]

kde = gaussian_kde(residuals)
x_kde = np.linspace(residuals.min(), residuals.max(), 200)
y_kde = kde(x_kde)

fig_residuals = go.Figure()
fig_residuals.add_trace(go.Bar(
    x=x_hist[:-1],
    y=y_hist,
    width=np.diff(x_hist),
    name='Histogram',
    marker=dict(color='lightblue'),
    opacity=0.6
))
fig_residuals.add_trace(go.Scatter(
    x=x_kde,
    y=y_kde * len(residuals) * np.diff(x_hist)[0],
    mode='lines',
    name='KDE',
    line=dict(color='darkblue')
))
fig_residuals.update_layout(
    title="Residual Distribution (Actual - Predicted)",
    xaxis_title="Residuals",
    yaxis_title="Frequency",
    bargap=0.05,
    height=400
)
st.plotly_chart(fig_residuals, use_container_width=True)

# -------------------------------
# Confidence Intervals
# -------------------------------
st.subheader("📊 Prediction Confidence Interval")

col3, col4 = st.columns(2)
confidence_level = col3.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
n_samples_to_show = col4.slider("Samples to Show", 10, 50, min(20, len(y_test_filtered)))

lower_bound, upper_bound = calculate_confidence_intervals(model, X_test, confidence=confidence_level)
indices_to_show = sorted(np.random.choice(len(y_test_filtered), n_samples_to_show, replace=False))

fig_ci = go.Figure()
fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=upper_bound[indices_to_show], fill=None, mode='lines'))
fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=lower_bound[indices_to_show], fill='tonexty', mode='lines',
                            name=f'{confidence_level*100:.0f}% Interval', fillcolor='rgba(0,100,80,0.2)'))
fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=y_pred_test_filtered[indices_to_show], mode='markers+lines', name='Predicted'))
fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=y_test_filtered.iloc[indices_to_show], mode='markers', name='Actual'))
fig_ci.update_layout(title='Prediction Confidence Interval', xaxis_title='Sample Index', yaxis_title='Value')
st.plotly_chart(fig_ci, use_container_width=True)

# -------------------------------
# SHAP
# -------------------------------
st.subheader("🔍 SHAP Interpretability")

def compute_shap(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X_train), explainer.shap_values(X_test)

try:
    explainer, shap_values_train, shap_values_test = compute_shap(model, X_train, X_test)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("SHAP Summary")
        fig1, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_test, X_test, show=False)
        st.pyplot(fig1)
        plt.close()

    with col6:
        st.subheader("SHAP Feature Impact")
        fig2, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
        st.pyplot(fig2)
        plt.close()

    # Single sample
    st.subheader("🎯 SHAP Waterfall (Single Sample)")
    col7, col8, col9 = st.columns([1, 2, 2])
    with col7:
        sample_idx = st.selectbox("Select Sample Index", list(range(min(20, len(X_test)))))
        sample = X_test.iloc[sample_idx]
        pred = y_pred_test_filtered[sample_idx]
        actual = y_test_filtered.iloc[sample_idx]
        
        st.write("**Sample Features**")
        st.dataframe(pd.DataFrame({'Feature': sample.index, 'Value': sample.values}))
        st.metric("Prediction", f"{pred:.3f}")
        st.metric("Actual", f"{actual:.3f}")
        st.metric("Error", f"{abs(pred - actual):.3f}")

    with col8:
        st.write("**SHAP Waterfall Plot**")
        fig3, ax = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values_test[sample_idx],
                             base_values=explainer.expected_value,
                             data=sample.values,
                             feature_names=sample.index.tolist()),
            show=False
        )
        st.pyplot(fig3)
        plt.close()

    with col9:
        try:
            payload = {
                "features": sample.values.astype(float).tolist(),
                'names': sample.index.tolist(),
                "shap": shap_values_test[sample_idx].astype(float).tolist(),
                "prediction": float(pred),
            }
            print('payload')
            print(payload)
            response = requests.post(
                "http://localhost:8000/explain_shap",
                json=payload
            )
            explanation = response.json()["explanation"]
            st.write("**GenAI Explanation**") 
            st.markdown(explanation, unsafe_allow_html=True)
            # copy_code = f"""
            #     <button onclick="navigator.clipboard.writeText(`{explanation}`); this.innerText='Copied!'">
            #         Copy to clipboard
            #     </button>
            # """
            # st.markdown(copy_code, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Explanation generation failed: {e}")

except Exception as e:
    st.error(f"SHAP computation failed: {e}")
    shap_values_test = None

# -------------------------------
# Download section
# -------------------------------
st.subheader("💾 Download Results")

col9, col10 = st.columns(2)
with col9:
    df_results = pd.DataFrame({
        "actual": y_test_filtered,
        "predicted": y_pred_test_filtered,
        "lower_bound": lower_bound[:len(y_test_filtered)],
        "upper_bound": upper_bound[:len(y_test_filtered)],
    })
    st.download_button("Download Predictions", df_results.to_csv(index=False), "predictions.csv", "text/csv")

with col10:
    if shap_values_test is not None:
        shap_df = pd.DataFrame(shap_values_test, columns=X_test.columns)
        st.download_button("Download SHAP Values", shap_df.to_csv(index=False), "shap_values.csv", "text/csv")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("This dashboard presents model performance, interpretability, and uncertainty analysis with group-level filters.")
