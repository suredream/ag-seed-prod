from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor
import numpy as np

from utils import load_Xy

# 使用前面已预处理好的数据
# X_features = X.drop(columns=['idx'], errors='ignore').astype(float)
# y_target = y

X, y =load_Xy()

# 定义参数搜索空间
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# 构建模型与 GridSearchCV
xgb = XGBRegressor()
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# 执行搜索
grid_search.fit(X, y)

# 最佳模型
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)

# 计算最终指标
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
rmse = root_mean_squared_error(y, y_pred)

# 输出结果
print({
    "Best Parameters": grid_search.best_params_,
    "MAE": round(mae, 4),
    "R2": round(r2, 4),
    "RMSE": round(rmse, 4)
})
