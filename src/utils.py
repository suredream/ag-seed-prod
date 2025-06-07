import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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


def model_eval(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):
    """
    Evaluate a regression model on both training and test data.
    
    Parameters:
        X_train, X_test: Features for train and test sets
        y_train, y_test: Targets for train and test sets
        
    Returns:
        metrics_dict: Dictionary containing MSE, MAE, R2 for train and test
    """

    # 指标计算
    metrics_dict = {
        "MSE": {
            "train": mean_squared_error(y_train, y_pred_train),
            "test": mean_squared_error(y_test, y_pred_test)
        },
        "MAE": {
            "train": mean_absolute_error(y_train, y_pred_train),
            "test": mean_absolute_error(y_test, y_pred_test)
        },
        "R2": {
            "train": r2_score(y_train, y_pred_train),
            "test": r2_score(y_test, y_pred_test)
        }
    }

    # 打印格式化表格
    print("{:<10} {:<15} {:<15}".format("Metric", "Train", "Test"))
    for metric, values in metrics_dict.items():
        print("{:<10} {:<15.4f} {:<15.4f}".format(metric, values["train"], values["test"]))
    print(X_train.columns)
    print(X_train.shape)
    return metrics_dict
