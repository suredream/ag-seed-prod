# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import toml
from sklearn.preprocessing import LabelEncoder
from src.utils import model_eval
label_encoder = LabelEncoder()

# === Load Config ===
config = toml.load("config/xgb.toml")

# === Utility Functions ===
def preprocess_data(df, config):
    df['PRODUCT_encoded'] = label_encoder.fit_transform(df['PRODUCT'])
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

def train_and_save(df, config):
    df = preprocess_data(df, config)
    df, scaler, pca = generate_pca_features(df, config)
    df_cat = df[config['categorical']['columns']]
    df = pd.get_dummies(df, columns=config['categorical']['columns'])
    df = pd.concat([df, df_cat], axis=1)

    X = df[config['features']['final']]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['UNITS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['STATE'])
    model = XGBRegressor(**config['xgb_params'])
    model.fit(X_train, y_train)


    # Save artifacts
    joblib.dump(model, config['output']['model_path'])
    joblib.dump(scaler, config['output']['scaler_path'])
    joblib.dump(pca, config['output']['pca_path'])
    joblib.dump((X_train, X_test, y_train, y_test), config['output']['features_path'])
    joblib.dump(df, config['output']['cat_path'])

    # Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    model_eval(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test)

# === Inference Function ===
def inference(input_df, config):
    model = joblib.load(config['output']['model_path'])
    scaler = joblib.load(config['output']['scaler_path'])
    pca = joblib.load(config['output']['pca_path'])

    df = preprocess_data(input_df.copy(), config)
    pca_features = config['features']['pca_inputs']
    scaled = scaler.transform(df[pca_features])
    pca_result = pca.transform(scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    df = encode_categorical(df, config)
    X = df[config['features']['final']]
    pred = model.predict(X)
    return pred, X
