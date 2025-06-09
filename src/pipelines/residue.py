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
    """
    Preprocesses the input DataFrame by filling missing values and engineering new features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Fill missing values in 'DROUGHT_TOLERANCE' using the mean for each 'LIFECYCLE' and 'STATE'
    df['DROUGHT_TOLERANCE'] = df.groupby(['LIFECYCLE', 'STATE'])['DROUGHT_TOLERANCE'] \
        .transform(lambda x: x.fillna(x.mean().round()))

    # Fill missing values in 'BRITTLE_STALK' and 'PLANT_HEIGHT' with values from the config
    df['BRITTLE_STALK'] = pd.to_numeric(df['BRITTLE_STALK'], errors='coerce').fillna(config['impute']['BRITTLE_STALK'])
    df['PLANT_HEIGHT'] = pd.to_numeric(df['PLANT_HEIGHT'], errors='coerce').fillna(config['impute']['PLANT_HEIGHT'])

    # Feature Engineering
    df['PROTECTION_SCORE'] = df['DISEASE_RESISTANCE'] + df['INSECT_RESISTANCE'] + df['PROTECTION']
    df['PRODUCT_AGE'] = df['SALESYEAR'] - df['RELEASE_YEAR']
    df['AGE_X_PROTECTION'] = df['PRODUCT_AGE'] * df['PROTECTION_SCORE']
    df['HEIGHT_X_MATURITY'] = df['PLANT_HEIGHT'] * df['RELATIVE_MATURITY']

    # Feature Engineering: PREVIOUS_UNITS
    df['UNITS_NORM_BY_PRODUCT'] = df.groupby('PRODUCT')['UNITS'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    df = df.sort_values(by=['STATE', 'PRODUCT', 'SALESYEAR'])
    df['PREVIOUS_UNITS'] = (
        df.groupby(['STATE', 'PRODUCT'])['UNITS']
        .shift(1)
    )
    df['PREVIOUS_UNITS'] = df['PREVIOUS_UNITS'].fillna(0)

    return df


def generate_pca_features(df, config):
    """
    Generates PCA features from the specified input features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        config (dict): Configuration parameters.

    Returns:
        tuple: The DataFrame with PCA features, the scaler, and the PCA object.
    """
    pca_features = config['features']['pca_inputs']
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    scaled = scaler.fit_transform(df[pca_features])
    pca_result = pca.fit_transform(scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    return df, scaler, pca


def encode_categorical(df, config):
    """
    Encodes categorical features using one-hot encoding.

    Args:
        df (pd.DataFrame): The input DataFrame.
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical features.
    """
    for col in config['categorical']['one_hot_columns_flat']:  # all possible One-Hot columns
        df[col] = 0
    for col in config['categorical']['columns']:
        val = df[col].iloc[0]  # assume batch has same category in example
        encoded_col = f"{col}_{val}"
        if encoded_col in df.columns:
            df.loc[:, encoded_col] = 1
    return df


def train_and_save_residual(df, config):
    """
    Trains a residual model and saves the model artifacts.

    Args:
        df (pd.DataFrame): The input DataFrame.
        config (dict): Configuration parameters.
    """
    df = preprocess_data(df, config)
    df, scaler, pca = generate_pca_features(df, config)
    df_cat = df[config['categorical']['columns']]
    df = pd.get_dummies(df, columns=config['categorical']['columns'])
    df = pd.concat([df, df_cat], axis=1)

    X = df[config['features']['final']]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['UNITS']

    # Split data
    X_train, X_test, y_train, y_test, prod_train, prod_test = train_test_split(
        X, y, df['PRODUCT'], test_size=0.2, random_state=42, stratify=df['STATE'])

    # Train base model
    print('xgb_params', config['xgb_params'])
    base_model = XGBRegressor(**config['xgb_params'])
    base_model.fit(X_train, y_train)

    # Predict and compute residuals
    y_base_pred_train = base_model.predict(X_train)
    residuals_train = y_train - y_base_pred_train

    # Encode product_name
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    product_encoded_train = ohe.fit_transform(prod_train.to_frame())
    product_encoded_test = ohe.transform(prod_test.to_frame())

    # Train dummy residual model
    residual_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    residual_model.fit(product_encoded_train, residuals_train)

    # Inference
    y_base_pred_test = base_model.predict(X_test)
    y_residual_pred_test = residual_model.predict(product_encoded_test)
    y_final_pred = y_base_pred_test + y_residual_pred_test
    # temporarily set negative predictions to zero
    y_final_pred = np.maximum(y_final_pred, 0)

    # Inference train dataset
    y_base_pred_train = base_model.predict(X_train)
    y_residual_pred_train = residual_model.predict(product_encoded_train)
    y_pred_train = y_base_pred_train + y_residual_pred_train
    # temporarily set negative predictions to zero
    y_pred_train = np.maximum(y_pred_train, 0)

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
    z_score = norm.ppf(0.975)

    # Use GradientBoostingRegressor to build a model for uncertainty estimation
    residual_var_model = GradientBoostingRegressor()
    residual_var_model.fit(product_encoded_train, residual_target)
    predicted_residual_std = residual_var_model.predict(product_encoded_test)

    lower_bound = y_final_pred - z_score * predicted_residual_std
    lower_bound = np.maximum(lower_bound, 0)
    upper_bound = y_final_pred + z_score * predicted_residual_std

    predicted_residual_std_train = residual_var_model.predict(product_encoded_train)

    lower_bound_train = y_pred_train - z_score * predicted_residual_std_train
    lower_bound_train = np.maximum(lower_bound_train, 0)
    upper_bound_train = y_pred_train + z_score * predicted_residual_std_train

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

    data_train = X_train.copy()
    data_train['pred'] = y_pred_train
    data_train['lower_bound'] = lower_bound_train
    data_train['upper_bound'] = upper_bound_train

    data_test = X_test.copy()
    data_test['pred'] = y_final_pred
    data_test['lower_bound'] = lower_bound
    data_test['upper_bound'] = upper_bound
    df_combined = pd.concat([data_train, data_test], axis=0, ignore_index=True)
    df_combined.to_csv('data/case_study_data_combined.csv', index=False)

    prediction_interval_df = pd.DataFrame({
        'y_pred': y_final_pred,
        'predicted_std': predicted_residual_std,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })


# # === Inference Function ===
def inference_residual(input_df, product_name, config):
    """
    Infers the output using the residual model.

    Args:
        input_df (pd.DataFrame): The input DataFrame.
        product_name (str): The product name.
        config (dict): Configuration parameters.

    Returns:
        dict: A dictionary containing the prediction and explanation.
    """
    base_model = joblib.load(config['output']['base_model_path'])
    scaler = joblib.load(config['output']['base_scaler_path'])
    residual_model = joblib.load(config['output']['residue_model_path'])
    product_encoder = joblib.load(config['output']['residue_ohe_path'])

    df = preprocess_data(input_df.copy(), config)
    df = encode_categorical(df, config)
    X = df[config['features']['final']]

    # Trait prediction
    y_base_pred = base_model.predict(X)

    # Product name one-hot encoding
    product_name_series = pd.Series({'product_name': product_name})
    product_encoded = product_encoder.transform(product_name_series.to_frame())

    # Residual prediction
    y_residual_pred = residual_model.predict(product_encoded)

    # Final prediction
    y_final_pred = y_base_pred + y_residual_pred

    pred = [float(y[0]) for y in (y_final_pred, y_base_pred, y_residual_pred)]
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X).astype(float).tolist()[0]
    return {
        'pred': pred[0],
        'explain': {'pred': pred,
                    'input': X.iloc[0].tolist(),
                    'shap_values': shap_values,
                    'feature_names': config['features']['final']}
    }
