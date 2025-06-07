import sys
import os

# Dynamically add project root (containing 'src') to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import streamlit as st
import pandas as pd
import numpy as np
import toml
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.utils import load_artifacts, calculate_confidence_intervals
from scipy.stats import gaussian_kde
import requests
import joblib
st.set_page_config(page_title="Model Analysis Dashboard", layout="wide")

# config = toml.load("config/xgb.toml")
config = toml.load("config/residual.toml")
# model, scaler, pca, X_train, X_test, y_train, y_test, y_base_test, y_residual_test = load_artifacts(config)
model, X_train, X_test, y_train, y_pred_train, y_test, y_final_pred, y_base_test, y_residual_test = joblib.load(config['output']['app_path'])
# print(X_test.columns)

# -------------------------------
# Group filter selectors
# -------------------------------
st.sidebar.header("🔎 SODA dashboard")

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

# # Predict
# X_test = X_test_filtered[config['features']['final']]
# y_pred_test_filtered = model.predict(X_test)
# y_pred_train = model.predict(X_train)
y_pred_test_filtered = y_final_pred[mask]
y_base_test_filtered = y_base_test[mask]
y_residual_filtered = y_residual_test[mask]

# -------------------------------
# Evaluation metrics
# -------------------------------
col1, col2, col22 = st.columns([1.5, 2, 2])

with col1:
    st.subheader("📈 Model Eval Metrics")
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)

    test_rmse = np.sqrt(mean_squared_error(y_test_filtered, y_pred_test_filtered))
    test_r2 = r2_score(y_test_filtered, y_pred_test_filtered)
    test_mae = mean_absolute_error(y_test_filtered, y_pred_test_filtered)

    # metrics_df = pd.DataFrame({
    #     'Metric': ['RMSE', 'R²', 'MAE'],
    #     'Train': [f'{train_rmse:.3f}', f'{train_r2:.3f}', f'{train_mae:.3f}'],
    #     'Test': [f'{test_rmse:.3f}', f'{test_r2:.3f}', f'{test_mae:.3f}']
    # })
    # st.dataframe(metrics_df, use_container_width=True)
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
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_test_filtered, y=y_pred_test_filtered, mode='markers', name='Test'))
    min_val = min(y_test_filtered.min(), y_pred_test_filtered.min())
    max_val = max(y_test_filtered.max(), y_pred_test_filtered.max())
    fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect'))
    fig_scatter.update_layout(title='Predicted vs Actual', xaxis_title='Actual', yaxis_title='Predicted', showlegend=False, font=dict(size=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    residuals = y_test_filtered - y_pred_test_filtered
    hist_data = np.histogram(residuals, bins=30)
    x_hist = hist_data[1]
    y_hist = hist_data[0]


with col22:
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

# # -------------------------------
# # Confidence Intervals
# # -------------------------------
# st.subheader("📊 Prediction Confidence Interval")

# col3, col4 = st.columns(2)
# confidence_level = col3.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
# n_samples_to_show = col4.slider("Samples to Show", 10, 50, min(20, len(y_test_filtered)))

# lower_bound, upper_bound = calculate_confidence_intervals(model, X_test, confidence=confidence_level)
# indices_to_show = sorted(np.random.choice(len(y_test_filtered), n_samples_to_show, replace=False))

# fig_ci = go.Figure()
# fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=upper_bound[indices_to_show], fill=None, mode='lines'))
# fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=lower_bound[indices_to_show], fill='tonexty', mode='lines',
#                             name=f'{confidence_level*100:.0f}% Interval', fillcolor='rgba(0,100,80,0.2)'))
# fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=y_pred_test_filtered[indices_to_show], mode='markers+lines', name='Predicted'))
# fig_ci.add_trace(go.Scatter(x=list(range(n_samples_to_show)), y=y_test_filtered.iloc[indices_to_show], mode='markers', name='Actual'))
# fig_ci.update_layout(title='Prediction Confidence Interval', xaxis_title='Sample Index', yaxis_title='Value')
# st.plotly_chart(fig_ci, use_container_width=True)


# -------------------------------
# SHAP
# -------------------------------


st.subheader("🔍 Feature Impact")

def compute_shap(model, X_train, X_test):
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X_train), explainer.shap_values(X_test)

try:
    X_train_fea = X_train.drop(columns=['PRODUCT', 'STATE', 'LIFECYCLE'])
    X_test_bounds = X_test.copy()
    X_test = X_test.drop(columns=['PRODUCT', 'STATE', 'LIFECYCLE','lower_bound', 'upper_bound'])
    explainer, shap_values_train, shap_values_test = compute_shap(model, X_train_fea, X_test)

    col5, col6 = st.columns(2)

    # with col6:
    #     st.subheader("SHAP Summary")
    #     fig1, ax = plt.subplots(figsize=(5, 3))
    #     shap.summary_plot(shap_values_test, X_test, show=False)
    #     st.pyplot(fig1)
    #     plt.close()

    with col5:
        # st.subheader("SHAP Feature Impact")
        fig2, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
        st.pyplot(fig2)
        plt.close()


    df_pred = pd.read_csv('data/case_study_data_combined.csv').groupby(['PRODUCT','STATE'])[['pred','lower_bound','upper_bound']].mean().reset_index()
    # st.dataframe(df_pred.head(), use_container_width=True)
    df_pred["predicted_yield"] = df_pred["pred"]

    pred_mask = (
        df_pred['PRODUCT'].isin(selected_products) &
        df_pred['STATE'].isin(selected_states)
    )

    st.subheader("🔍 Top K Product")
    top_k = st.slider("", 5, 20, 5)
    ranking = df_pred[pred_mask].sort_values("predicted_yield", ascending=False).head(top_k)
    ranking_sorted = ranking.sort_values("predicted_yield", ascending=True)
    # fig = go.Figure(go.Bar(
    #     x=ranking_sorted["predicted_yield"],
    #     y=ranking_sorted["PRODUCT"],
    #     orientation='h',
    #     text=ranking_sorted["predicted_yield"].round(2),
    #     textposition="auto"
    # ))
    fig = go.Figure(go.Bar(
        x=ranking_sorted["predicted_yield"],
        y=ranking_sorted["PRODUCT"],
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
    # st.dataframe(ranking_sorted, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)


    # Single sample
    st.subheader("🎯 Single Prediction & Explanation")
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

except Exception as e:
    st.error(f"SHAP computation failed: {e}")
    shap_values_test = None

# # -------------------------------
# # Download section
# # -------------------------------
# st.subheader("💾 Download Results")

# col9, col10 = st.columns(2)
# with col9:
#     df_results = pd.DataFrame({
#         "actual": y_test_filtered,
#         "predicted": y_pred_test_filtered,
#         "lower_bound": lower_bound[:len(y_test_filtered)],
#         "upper_bound": upper_bound[:len(y_test_filtered)],
#     })
#     st.download_button("Download Predictions", df_results.to_csv(index=False), "predictions.csv", "text/csv")

# with col10:
#     if shap_values_test is not None:
#         shap_df = pd.DataFrame(shap_values_test, columns=X_test.columns)
#         st.download_button("Download SHAP Values", shap_df.to_csv(index=False), "shap_values.csv", "text/csv")


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("This dashboard presents model performance, interpretability, and uncertainty analysis with group-level filters.")
