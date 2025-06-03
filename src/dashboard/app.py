import sys
import os

# Dynamically add project root (containing 'src') to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import toml
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from model_utils import load_artifacts, calculate_confidence_intervals
from xgboost import XGBRegressor

def load_artifacts(config):
    model = joblib.load(config['output']['model_path'])
    scaler = joblib.load(config['output']['scaler_path'])
    pca = joblib.load(config['output']['pca_path'])
    X_train, X_test, y_train, y_test = joblib.load(config['output']['features_path'])
    return model, scaler, pca, X_train, X_test, y_train, y_test

def calculate_confidence_intervals(model, X_test, n_bootstrap=100, confidence=0.95):
    predictions = []
    n_samples = len(X_test)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X_test.iloc[indices]
        pred = model.predict(X_bootstrap)
        predictions.append(pred)

    predictions = np.array(predictions)
    lower = np.percentile(predictions, (1-confidence)/2*100, axis=0)
    upper = np.percentile(predictions, (1+(confidence))/2*100, axis=0)
    return lower, upper


st.set_page_config(page_title="Model Analysis Dashboard", layout="wide")

# -------------------------------
# Load config and model/data
# -------------------------------
config = toml.load("config/xgb.toml")
model, scaler, pca, X_train, X_test, y_train, y_test = load_artifacts(config)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# -------------------------------
# Evaluation metrics
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Evaluation Metrics")
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'R²', 'MAE'],
        'Train': [f'{train_rmse:.3f}', f'{train_r2:.3f}', f'{train_mae:.3f}'],
        'Test': [f'{test_rmse:.3f}', f'{test_r2:.3f}', f'{test_mae:.3f}']
    })
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("🎯 Feature Importance")

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    fig_importance = px.bar(importance_df, x='importance', y='feature', orientation='h')
    fig_importance.update_layout(title='Feature Importance', height=400)
    st.plotly_chart(fig_importance, use_container_width=True)



with col2:
    from scipy.stats import gaussian_kde

    # Compute residuals
    residuals = y_test - y_pred_test

    # Create histogram
    hist_data = np.histogram(residuals, bins=30)
    x_hist = hist_data[1]
    y_hist = hist_data[0]

    # Compute KDE using scipy
    kde = gaussian_kde(residuals)
    x_kde = np.linspace(residuals.min(), residuals.max(), 200)
    y_kde = kde(x_kde)

    # Build figure
    fig_residuals = go.Figure()

    # Histogram
    fig_residuals.add_trace(go.Bar(
        x=x_hist[:-1],
        y=y_hist,
        width=np.diff(x_hist),
        name='Histogram',
        marker=dict(color='lightblue'),
        opacity=0.6
    ))

    # KDE line
    fig_residuals.add_trace(go.Scatter(
        x=x_kde,
        y=y_kde * len(residuals) * np.diff(x_hist)[0],  # scale to match histogram height
        mode='lines',
        name='KDE',
        line=dict(color='darkblue')
    ))

    # Layout
    fig_residuals.update_layout(
        title="Residual Distribution (Actual - Predicted)",
        xaxis_title="Residuals",
        yaxis_title="Frequency",
        bargap=0.05,
        height=400
    )

    st.plotly_chart(fig_residuals, use_container_width=True)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode='markers', name='Test'))
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect'))
    fig_scatter.update_layout(title='Predicted vs Actual', xaxis_title='Actual', yaxis_title='Predicted')
    st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------
# Confidence Interval Analysis
# -------------------------------
st.subheader("📊 Prediction Confidence Intervals")
col3, col4 = st.columns(2)
confidence_level = col3.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
n_samples_to_show = col4.slider("Number of Samples to Show", 10, 50, 20)

lower_bound, upper_bound = calculate_confidence_intervals(model, X_test, confidence=confidence_level)
indices_to_show = sorted(np.random.choice(len(y_test), n_samples_to_show, replace=False))

fig_ci = go.Figure()
fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=upper_bound[indices_to_show], fill=None, mode='lines'))
fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=lower_bound[indices_to_show], fill='tonexty', mode='lines',
                            name=f'{confidence_level*100:.0f}% Interval', fillcolor='rgba(0,100,80,0.2)'))
fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=y_pred_test[indices_to_show], mode='markers+lines', name='Predicted'))
fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=y_test.iloc[indices_to_show], mode='markers', name='Actual'))

fig_ci.update_layout(title='Prediction Confidence Interval', xaxis_title='Sample Index', yaxis_title='Value')
st.plotly_chart(fig_ci, use_container_width=True)

# -------------------------------
# SHAP Explanation
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

    # Single Prediction Explanation
    st.subheader("🎯 SHAP Waterfall for Single Sample")
    sample_idx = st.selectbox("Select Sample Index", list(range(min(20, len(X_test)))))
    sample = X_test.iloc[sample_idx]
    pred = y_pred_test[sample_idx]
    actual = y_test.iloc[sample_idx]

    col7, col8 = st.columns(2)
    with col7:
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

except Exception as e:
    st.error(f"SHAP computation failed: {e}")
    shap_values_test = None  # Ensure it's defined

# -------------------------------
# Downloads
# -------------------------------
st.subheader("💾 Download Results")

col9, col10 = st.columns(2)
with col9:
    df_results = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred_test,
        "lower_bound": lower_bound[:len(y_test)],
        "upper_bound": upper_bound[:len(y_test)],
    })
    st.download_button("Download Predictions", df_results.to_csv(index=False), "predictions.csv", "text/csv")

with col10:
    if shap_values_test is not None:
        shap_df = pd.DataFrame(shap_values_test, columns=X_test.columns)
        st.download_button("Download SHAP Values", shap_df.to_csv(index=False), "shap_values.csv", "text/csv")
    else:
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        st.download_button("Download Feature Importance", importance_df.to_csv(index=False), "feature_importance.csv", "text/csv")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("This dashboard presents comprehensive analysis of the model, including performance evaluation, confidence intervals, and SHAP-based interpretability.")
