我正在构建一个用于农业种子产品销量预测的 Streamlit 仪表板应用，请帮我实现以下功能：

🎯 项目目标：
- 使用多个机器学习模型（如 XGBoost、LightGBM、线性回归）对销量进行预测
- 比较这些模型的预测性能（如 MAPE、RMSE、Forecast Bias）
- 展示每个模型的预测结果和主导因素（Drivers）

📦 功能需求：

1. 创建一个 `MODEL_REGISTRY`，用于注册不同的模型及其构造方法。例如支持 "XGBoost"、"LightGBM"、"LinearRegression"

2. 实现一个通用的 `run_pipeline(model_name, model_constructor, X, y)` 函数，完成以下任务：
    - 拟合模型
    - 生成预测值（强制非负）
    - 计算指标（MAPE, RMSE, Bias）
    - 调用统一解释接口生成每条预测的驱动因子说明（Drivers）

3. 输出结构：
    - 每个模型的预测结果为一个 DataFrame（含：预测值、置信区间、Drivers）
    - 每个模型的指标为一个 dict（含：MAPE, RMSE, Bias）

4. 所有模型输出统一存入一个结构，如：
```python
model_results = {
    "XGBoost": {"predictions": ..., "metrics": ..., "drivers": [...]},
    "LightGBM": {...},
}
