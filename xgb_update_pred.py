import pandas as pd
import toml
import os
from src.pipelines.xgb import train_and_save, inference  # 确保该模块已保存为 train_model.py
from src.utils import load_artifacts, calculate_confidence_intervals

# # === 加载配置文件 ===
config = toml.load("config/xgb.toml")
# model, scaler, pca, X_train, X_test, y_train, y_test = load_artifacts(config)

# print(X_test.columns)

# # === 训练模型 ===
if not os.path.exists(config['output']['model_path']):
# if True:
    data_path = config['input']['data_path']
    df = pd.read_csv(data_path)
    train_and_save(df, config)
    print("✅ 模型训练完成，已保存模型与PCA/Scaler 文件。")


# === 示例输入：来自一个 dict ===
input_dict = {
    'PRODUCT': 'P123',
    'SALESYEAR': 2024,
    'RELEASE_YEAR': 2020,
    'DISEASE_RESISTANCE': 3,
    'INSECT_RESISTANCE': 4,
    'PROTECTION': 2,
    'DROUGHT_TOLERANCE': 5.0,
    'BRITTLE_STALK': None,        # 可缺失
    'PLANT_HEIGHT': 6.0,
    'RELATIVE_MATURITY': 3,
    'STATE': 'Texas',
    'LIFECYCLE': 'EXPANSION'
}

# === 转为 DataFrame ===
input_df = pd.DataFrame([input_dict])

# === 调用推理函数 ===
prediction = inference(input_df, config)

print(f"🔮 预测销量（UNITS）为: {prediction[0]:.2f}")
