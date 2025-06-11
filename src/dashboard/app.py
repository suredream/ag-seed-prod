import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import toml
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import joblib
st.set_page_config(page_title="Model Analysis Dashboard", layout="wide")

def compute_shap(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X_train), explainer.shap_values(X_test)


config = toml.load("config/residual.toml")
model, X_train, X_test, y_train, y_pred_train, y_test, y_final_pred, y_base_test, y_residual_test = joblib.load(config['output']['app_path'])

# Sidebar: Group filter selectors
st.sidebar.header("🔎 SODA Dashboard")

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

# Multiselect filters for STATE, LIFECYCLE and PRODUCT
selected_states = st.sidebar.multiselect("Select STATE(s)", state_options, default=state_options)
selected_lifecycles = st.sidebar.multiselect("Select LIFECYCLE(s)", lifecycle_options, default=lifecycle_options)
selected_products = st.sidebar.multiselect("Select PRODUCT(s)", product_options, default=product_options)

# Filter data based on selected options
mask = (
    X_test['PRODUCT'].isin(selected_products) &
    X_test['STATE'].isin(selected_states) &
    X_test['LIFECYCLE'].isin(selected_lifecycles)
)
X_test_filtered = X_test[mask]
y_test_filtered = y_test[mask]

# Stop if no data is selected
if len(X_test_filtered) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# Get predictions for filtered data
y_pred_test_filtered = y_final_pred[mask]
y_base_test_filtered = y_base_test[mask]
y_residual_filtered = y_residual_test[mask]

# Main Panel: Evaluation metrics
col1, col2, col22 = st.columns([1.5, 2, 2])

with col1:
    st.subheader("📈 [Internal] Model Metrics")
    # Calculate evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_test_filtered))
    test_r2 = r2_score(y_test_filtered, y_pred_test_filtered)
    test_mae = mean_absolute_error(y_test_filtered, y_pred_test_filtered)

    # Display Train metrics
    train_col1, train_col2, train_col3 = st.columns(3)
    test_col1, test_col2, test_col3 = st.columns(3)
    with train_col1:
        st.metric(label="Train RMSE", value=f"{train_rmse:.3f}")
    with train_col2:
        st.metric(label="Train R²", value=f"{train_r2:.3f}")
    with train_col3:
        st.metric(label="Train MAE", value=f"{train_mae:.3f}")

    # Display Test metrics
    with test_col1:
        st.metric(label="Test RMSE", value=f"{test_rmse:.3f}")
    with test_col2:
        st.metric(label="Test R²", value=f"{test_r2:.3f}")
    with test_col3:
        st.metric(label="Test MAE", value=f"{test_mae:.3f}")

with col2:
    # Scatter plot of predicted vs actual values
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_test_filtered, y=y_pred_test_filtered, mode='markers', name='Test'))
    min_val = min(y_test_filtered.min(), y_pred_test_filtered.min())
    max_val = max(y_test_filtered.max(), y_pred_test_filtered.max())
    fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect'))
    fig_scatter.update_layout(title='Predicted vs Actual', xaxis_title='Actual', yaxis_title='Predicted', showlegend=False, font=dict(size=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Calculate residuals
    residuals = y_test_filtered - y_pred_test_filtered
    hist_data = np.histogram(residuals, bins=30)
    x_hist = hist_data[1]
    y_hist = hist_data[0]


with col22:
    # KDE plot of residuals
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
        title="Residual Distribution",
        xaxis_title="Residuals",
        yaxis_title="Frequency",
        showlegend=False,
        bargap=0.05,
        height=400
    )
    st.plotly_chart(fig_residuals, use_container_width=True)

st.subheader("🔍 [Internal] Feature Impact")
try:
    # Prepare data for SHAP computation
    X_train_fea = X_train.drop(columns=['PRODUCT', 'STATE', 'LIFECYCLE'])
    X_test_bounds = X_test.copy()
    X_test = X_test.drop(columns=['PRODUCT', 'STATE', 'LIFECYCLE','lower_bound', 'upper_bound'])
    explainer, shap_values_train, shap_values_test = compute_shap(model, X_train_fea, X_test)

    col5, col6 = st.columns(2)

    with col5:
        # SHAP Feature Impact Plot
        fig2, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
        st.pyplot(fig2)
        plt.close()


    df_pred = pd.read_csv('data/case_study_data_combined.csv').groupby(['PRODUCT','STATE'])[['pred','lower_bound','upper_bound']].mean().reset_index()
    df_pred["predicted_yield"] = df_pred["pred"]

    pred_mask = (
        df_pred['PRODUCT'].isin(selected_products) &
        df_pred['STATE'].isin(selected_states)
    )

    st.subheader("🔍 [StakeholderTool] Top Products")
    top_k = st.slider("", 5, 20, 5)
    ranking = df_pred[pred_mask].sort_values("predicted_yield", ascending=False).head(top_k)
    ranking_sorted = ranking.sort_values("predicted_yield", ascending=True)
    ranking_sorted['PRODUCT_REGION'] = ranking['PRODUCT'] + '_' + ranking['STATE']
    fig = go.Figure(go.Bar(
        x=ranking_sorted["predicted_yield"],
        y=ranking_sorted["PRODUCT_REGION"],
        orientation='h',
        text=ranking_sorted["predicted_yield"].round(2),
        textposition="auto",
        error_x=dict(
            type='data',
            symmetric=False,
            array=ranking_sorted["upper_bound"] - ranking_sorted["predicted_yield"],
            arrayminus=ranking_sorted["predicted_yield"] - ranking_sorted["lower_bound"]
        )
    ))
    fig.update_layout(
        title=f"Top {top_k} Product Predictions",
        xaxis_title="Predicted Seed Production",
        yaxis_title="PRODUCT",
        height=400,
        margin=dict(l=100, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


    # Single sample
    st.subheader("🎯 [StakeholderTool]Prediction & Explanation")
    col7, col8, col9 = st.columns([1, 2, 1])
    with col7:
        product_options = X_test_bounds['PRODUCT'].unique().tolist()
        sample_select = st.selectbox("Select Sample Product", product_options)
        sample_idx = product_options.index(sample_select)
        sample = X_test.iloc[sample_idx]
        pred = y_pred_test_filtered[sample_idx]
        pred1 = y_base_test_filtered[sample_idx]
        pred2 = y_residual_filtered[sample_idx]
        actual = y_test_filtered.iloc[sample_idx]
        uplimit = X_test_bounds['upper_bound'].iloc[sample_idx]
        lowlimit = X_test_bounds['lower_bound'].iloc[sample_idx]

        st.write("**Sample Features**")
        st.dataframe(pd.DataFrame({'Feature': sample.index, 'Value': sample.values}))
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", f"{pred:.3f}")
        c2.metric("Lower", f"{lowlimit:.3f}")
        c3.metric("Upper", f"{uplimit:.3f}")

    with col9:
        st.write("**SHAP Waterfall Plot**")
        fig3, ax = plt.subplots(figsize=(10, 15))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values_test[sample_idx],
                             base_values=explainer.expected_value,
                             data=sample.values,
                             feature_names=sample.index.tolist()),
            show=False
        )
        st.pyplot(fig3)
        plt.close()


    with col8:
        try:
            payload = {
                "features": sample.values.astype(float).tolist(),
                'names': sample.index.tolist(),
                "shap": shap_values_test[sample_idx].astype(float).tolist(),
                "prediction": [float(y) for y in [pred, pred1, pred2]],
            }
            response = requests.post(
                "http://localhost:8000/explain_shap",
                json=payload
            )
            explanation = response.json()["explanation"]
            st.write("**GenAI Explanation**")
            st.markdown(explanation)
        except Exception as e:
            st.error(f"Explanation generation failed: {e}")

        # action icon
        icon_bar = """
        <style>
        .icon-bar {
            display: flex;
            gap: 0px; /* 无间距 */
        }
        .icon-button {
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            padding: 6px;
        }
        .icon-button:hover {
            background-color: #eee;
            border-radius: 5px;
        }
        </style>

        <div class="icon-bar">
            <button class="icon-button" title="Copy">📋</button>
            <button class="icon-button" title="Good">👍</button>
            <button class="icon-button" title="Bad">👎</button>
            <button class="icon-button" title="Audio">🔊</button>
            <button class="icon-button" title="Add to Favorites">➕</button>
            <button class="icon-button" title="Comment">✏️</button>
        </div>
        """
        st.markdown(icon_bar, unsafe_allow_html=True)

except Exception as e:
    st.error(f"SHAP computation failed: {e}")
    shap_values_test = None

# Footer
st.markdown("---")
st.markdown("This dashboard presents model performance, interpretability, and uncertainty analysis with group-level filters.")
