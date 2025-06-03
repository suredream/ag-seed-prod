import pandas as pd
import sqlite3
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from utils import compute_drivers
import numpy as np

df_raw = pd.read_csv('case_study_data.csv')

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
def preprocess_data(df):
    df['Years_Since_Release'] = df['SALESYEAR'] - df['RELEASE_YEAR']

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['STATE', 'LIFECYCLE'])
    ordinal_cols = ['DROUGHT_TOLERANCE', 'BRITTLE_STALK', 'PLANT_HEIGHT', 'RELATIVE_MATURITY']
    scaler = MinMaxScaler()
    df_encoded[ordinal_cols] = scaler.fit_transform(df_encoded[ordinal_cols])
    df_encoded = df_encoded.drop(columns=['PRODUCT', 'RELEASE_YEAR'])
    if 'Lifecycle Stage_Phaseout' in df_encoded.columns:
        df_encoded = df_encoded[df_encoded['Lifecycle Stage_Phaseout'] == 0]
        df_encoded = df_encoded.drop(columns=['Lifecycle Stage_Phaseout'])
    return df_encoded

df_raw['idx'] = df_raw.index  # Save original index
df_encoded = preprocess_data(df_raw)

# Train XGBRegressor and get predictions
from xgboost import XGBRegressor
X = df_encoded.drop(columns=['UNITS','idx'])
# print(X.dtypes) 
y = df_encoded['UNITS']
model = XGBRegressor()
model.fit(X, y)
y_pred = model.predict(X)
y_pred = np.maximum(y_pred, 0)

# print('Y values', y[:50])
# print('Y pred values', y_pred[:59])
# print('extrea Y', np.abs((y - y_pred)/y).max() )

# Compute metrics
y_safe = np.where(np.isclose(y, 0), 1e-6, y)
mae = mean_absolute_error(y, y_pred)
mape = mean_absolute_percentage_error(y_safe, y_pred)
rmse = root_mean_squared_error(y, y_pred)
forecast_bias_y = ((y_pred - y) / y)
forecast_bias_y[~np.isfinite(forecast_bias_y)] = np.nan
forecast_bias = np.nanmean(forecast_bias_y)
# .mean()

# Inventory metrics (placeholders as they require domain-specific thresholds)
stockout_rate_reduction = "N/A (requires actual stockout labels)"
inventory_turnover_ratio = "N/A (requires inventory volume and COGS)"
days_inventory_outstanding = "N/A (requires average inventory & daily COGS)"


result_metric = {
    "MAE": round(mae, 4),
    "RMSE": round(rmse, 4),
    "Forecast Bias": round(forecast_bias, 4),
    "Stockout Rate Reduction": stockout_rate_reduction,
    "Inventory Turnover Ratio": inventory_turnover_ratio,
    "Days Inventory Outstanding": days_inventory_outstanding
}
print(result_metric)

X_clean = X.astype(float) # bool cann't be used in TreeExplainer compute
# check if there are any non-numeric columns
assert X_clean.select_dtypes(include='object').empty, "X still contains non-numeric columns!"

# Create a prediction DataFrame with original indices
df_pred = pd.DataFrame({
    'idx': X.index,
    'Units_Pred': y_pred,
    'Lower_Bound': y_pred * 0.9,
    'Upper_Bound': y_pred * 1.1,
    'Drivers': compute_drivers(X, model, top_n=3)
})
# Merge prediction back to original data using index
df_final = pd.merge(df_raw, df_pred, on='idx', how='left').drop(columns=['idx'])

# Save to SQLite for Streamlit app use
conn = sqlite3.connect("forecast_demo.db")
df_final.to_sql("forecasts", conn, if_exists="replace", index=False)
conn.commit()
conn.close()