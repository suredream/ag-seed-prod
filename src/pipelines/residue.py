# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import toml
import shap
from src.utils import model_eval
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm

# === Load Config ===
config = toml.load("config/residual.toml")

# === Utility Functions ===
def preprocess_data(df, config):
    # df['PRODUCT_encoded'] = label_encoder.fit_transform(df['PRODUCT'])
    # Fill missing values
    df['DROUGHT_TOLERANCE'] = df.groupby(['LIFECYCLE', 'STATE'])['DROUGHT_TOLERANCE']\
        .transform(lambda x: x.fillna(x.mean().round()))

    df['BRITTLE_STALK'] = pd.to_numeric(df['BRITTLE_STALK'], errors='coerce').fillna(config['impute']['BRITTLE_STALK'])
    df['PLANT_HEIGHT'] = pd.to_numeric(df['PLANT_HEIGHT'], errors='coerce').fillna(config['impute']['PLANT_HEIGHT'])

    # Feature Engineering
    df['PROTECTION_SCORE'] = df['DISEASE_RESISTANCE'] + df['INSECT_RESISTANCE'] + df['PROTECTION']
    df['PRODUCT_AGE'] = df['SALESYEAR'] - df['RELEASE_YEAR']
    df['AGE_X_PROTECTION'] = df['PRODUCT_AGE'] * df['PROTECTION_SCORE']
    df['HEIGHT_X_MATURITY'] = df['PLANT_HEIGHT'] * df['RELATIVE_MATURITY']
    df['IS_NEW_PRODUCT'] = (df['PRODUCT_AGE'] <= 2).astype(int)


    # # 添加上一年的 UNITS（按 STATE+PRODUCT 分组后向下移动一行）
    df['UNITS_NORM_BY_PRODUCT'] = df.groupby('PRODUCT')['UNITS'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    df = df.sort_values(by=['STATE', 'PRODUCT', 'SALESYEAR'])
    df['PREVIOUS_UNITS'] = (
        df.groupby(['STATE', 'PRODUCT'])['UNITS']
        .shift(1)
    )
    df['PREVIOUS_UNITS'] = df['PREVIOUS_UNITS'].fillna(0)

    return df

def generate_pca_features(df, config):
    pca_features = config['features']['pca_inputs']
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    scaled = scaler.fit_transform(df[pca_features])
    pca_result = pca.fit_transform(scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    return df, scaler, pca

def encode_categorical(df, config):
    for col in config['categorical']['one_hot_columns_flat']:  # all possible One-Hot columns
        df[col] = 0
    for col in config['categorical']['columns']:
        val = df[col].iloc[0]  # assume batch has same category in example
        encoded_col = f"{col}_{val}"
        if encoded_col in df.columns:
            df.loc[:, encoded_col] = 1
    return df

def train_and_save_residual(df, config):
    df = preprocess_data(df, config)
    # print(df.columns.to_list())
    df, scaler, pca = generate_pca_features(df, config)
    # df = encode_categorical(df, config)
    df_cat = df[config['categorical']['columns']]
    df = pd.get_dummies(df, columns=config['categorical']['columns'])
    df = pd.concat([df, df_cat], axis=1)

    X = df[config['features']['final']]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    # y = df[config['target']['column']]#.squeeze()
    y = df['UNITS']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split data
    X_train, X_test, y_train, y_test, prod_train, prod_test = train_test_split(
        X, y, df['PRODUCT'], test_size=0.2, random_state=42, stratify=df['STATE'])

    # Train base model
    # base_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    base_model = XGBRegressor(**config['xgb_params'])
    base_model.fit(X_train, y_train)

    # Predict and compute residuals
    y_base_pred_train = base_model.predict(X_train)
    residuals_train = y_train - y_base_pred_train

    # Encode product_name
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    product_encoded_train = ohe.fit_transform(prod_train.to_frame())
    # print(product_encoded_train.shape, product_encoded_train)
    product_encoded_test = ohe.transform(prod_test.to_frame())

    # # Train residual model
    residual_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    residual_model.fit(product_encoded_train, residuals_train)

    # # Inference
    y_base_pred_test = base_model.predict(X_test)
    y_residual_pred_test = residual_model.predict(product_encoded_test)
    y_final_pred = y_base_pred_test + y_residual_pred_test

    # # Inference train dataset
    y_base_pred_train = base_model.predict(X_train)
    y_residual_pred_train = residual_model.predict(product_encoded_train)
    y_pred_train = y_base_pred_train + y_residual_pred_train

    # Compute metrics
    base_mse = mean_squared_error(y_test, y_base_pred_test)
    base_mae = mean_absolute_error(y_test, y_base_pred_test)
    base_r2 = r2_score(y_test, y_base_pred_test)

    print(base_mse, base_mae, base_r2)

    final_mse = mean_squared_error(y_test, y_final_pred)
    final_mae = mean_absolute_error(y_test, y_final_pred)
    final_r2 = r2_score(y_test, y_final_pred)

    # Uncertainty
    residual_target = np.abs(residuals_train)

    # 特征输入可以使用与 residual_model 相同的 product_encoded_train 或 X_train
    residual_var_model = GradientBoostingRegressor()
    residual_var_model.fit(product_encoded_train, residual_target)
    predicted_residual_std = residual_var_model.predict(product_encoded_test)
    z_score = norm.ppf(0.975)
    lower_bound = y_final_pred - z_score * predicted_residual_std
    lower_bound = np.maximum(lower_bound, 0)
    upper_bound = y_final_pred + z_score * predicted_residual_std


    print(final_mse, final_mae, final_r2)
    joblib.dump(base_model, config['output']['base_model_path'])
    joblib.dump(scaler, config['output']['base_scaler_path'])
    joblib.dump(ohe, config['output']['residue_ohe_path'])
    joblib.dump(residual_model, config['output']['residue_model_path'])

    X_train = pd.concat([X_train, df_cat], axis=1, join='inner')
    X_test = pd.concat([X_test, df_cat], axis=1, join='inner')
    X_test['lower_bound'] = lower_bound
    X_test['upper_bound'] = upper_bound

    joblib.dump((base_model, X_train, X_test, y_train, y_pred_train, y_test, y_final_pred, y_base_pred_test, y_residual_pred_test), config['output']['app_path'])

    # Evaluation
    model_eval(X_train, X_test, y_train, y_test, y_pred_train, y_final_pred)



    # ------------------------
    # Step 4: 返回预测区间结果
    # ------------------------

    prediction_interval_df = pd.DataFrame({
        'y_pred': y_final_pred,
        'predicted_std': predicted_residual_std,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })

    print(prediction_interval_df.shape)

# # === Inference Function ===
def inference_residual(input_df, product_name, config):
    base_model = joblib.load(config['output']['base_model_path'])
    scaler = joblib.load(config['output']['base_scaler_path'])
    residual_model = joblib.load(config['output']['residue_model_path'])
    product_encoder = joblib.load(config['output']['residue_ohe_path'])
#     pca = joblib.load(config['output']['pca_path'])

    df = preprocess_data(input_df.copy(), config)
#     pca_features = config['features']['pca_inputs']
#     scaled = scaler.transform(df[pca_features])
#     pca_result = pca.transform(scaled)
#     df['PCA1'] = pca_result[:, 0]
#     df['PCA2'] = pca_result[:, 1]
    df = encode_categorical(df, config)
    X = df[config['features']['final']]
#     pred = model.predict(X)
#     return pred, X

    # X_trait = trait_encoder(input_df)
    # Trait prediction
    y_base_pred = base_model.predict(X)

    # Product name one-hot encoding
    product_name_series = pd.Series({'product_name': product_name})
    product_encoded = product_encoder.transform(product_name_series.to_frame())

    # Residual prediction
    y_residual_pred = residual_model.predict(product_encoded)

    # Final prediction
    y_final_pred = y_base_pred + y_residual_pred
    # print('y_final_pred', y_final_pred, y_base_pred, y_residual_pred)

    pred = [float(y[0]) for y in (y_final_pred, y_base_pred, y_residual_pred)]
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X).astype(float).tolist()[0]
    return {
        'pred': pred[0],
        'explain': {'pred': pred,
                    'input': X.iloc[0].tolist(),
                    'shap_values':shap_values,
                    'feature_names': config['features']['final']}
    }
