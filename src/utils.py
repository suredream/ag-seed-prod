import joblib
import numpy as np


def load_artifacts(config):
    model = joblib.load(config['output']['model_path'])
    scaler = joblib.load(config['output']['scaler_path'])
    pca = joblib.load(config['output']['pca_path'])

    data = joblib.load(config['output']['cat_path'])
    ids_df = data[config['features']['id_columns']]
    X_train, X_test, y_train, y_test = joblib.load(config['output']['features_path'])
    X_test = X_test.join(ids_df, how='left')
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