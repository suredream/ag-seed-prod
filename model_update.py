import pandas as pd
import toml
import os
import argparse
from src.pipelines.xgb import train_and_save, inference  # 确保该模块已保存为 train_model.py
from src.pipelines.residue import train_and_save_residual, inference_residual  # 确保该模块已保存为 train_model.py
from src.pipelines.explain import call_explain, build_business_prompt, build_prompt  # 确保该模块已保存为 train_model.py
# from src.utils import load_artifacts, calculate_confidence_intervals
import shap
import joblib

# === 命令行参数解析 ===
parser = argparse.ArgumentParser(description='Train and predict model.')
parser.add_argument('--model', type=str, default='residual', choices=['residual', 'xgb'], help='Specify the model type (residual or xgb).')
parser.add_argument('--pred_only', default=False, action='store_false', help='Only perform prediction, skip training.')
args = parser.parse_args()
# print(args)

# === 加载配置文件 ===
model_type = args.model
if model_type == 'residual':
    config = toml.load("config/residual.toml")
    model_dict = {'train_and_save': train_and_save_residual, 'inference': inference_residual}
elif model_type == 'xgb':
    config = toml.load("config/xgb.toml")
    model_dict = {'train_and_save': train_and_save, 'inference': inference}
else:
    raise ValueError("Invalid model type. Choose 'residual' or 'xgb'.")

# === 训练模型 ===
if args.pred_only:
# if not os.path.exists(config['output']['model_path']):
    data_path = config['input']['data_path']
    df = pd.read_csv(data_path)
    model_dict['train_and_save'](df, config)
    print(f"✅ {model_type} 模型训练完成，已保存模型与PCA/Scaler 文件。")


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

base_model = joblib.load(config['output']['base_model_path'])

# # === 转为 DataFrame ===
input_df = pd.DataFrame([input_dict])
pred_dict = model_dict['inference'](input_df, input_dict['PRODUCT'], config)
explain = pred_dict['explain']
prompt = build_prompt(explain['input'], explain['feature_names'], explain['shap_values'], explain['pred'])
print(call_explain(prompt))
# # === 调用推理函数 ===
# y_final_pred, y_base_pred, y_residual_pred, X = model_dict['inference'](input_df, input_dict['PRODUCT'], config)
# print(y_final_pred, y_base_pred, y_residual_pred)
# print(type(y_final_pred), type(y_base_pred), type(y_residual_pred))
# print(f"🔮 预测销量（UNITS）为: {prediction[0]:.2f}")
# Example usage
# example_prompt = build_business_prompt(
#     product_id="P43",
#     shap_summary=[
#         {"feature": "Drought_Tolerance", "value": 5, "impact": 20.1},
#         {"feature": "Plant_Height", "value": 2, "impact": -15.7},
#         {"feature": "Lifecycle_Stage_Expansion", "value": 1, "impact": 10.3},
#     ],
#     prediction_context={
#         "trait_model_output": 110.5,
#         "residual_model_output": 39.7,
#         "predicted_units": 150.2
#     }
# )

# print(call_explain(example_prompt))

# explainer = shap.TreeExplainer(base_model)
# shap_values = explainer.shap_values(X).astype(float).tolist()[0]
# # print(float(prediction[0]))
# print(X.iloc[0].tolist(), config['features']['final'], shap_values, float(prediction[0]))
# print(len(X.iloc[0].tolist()), len(config['features']['final']), len(shap_values))
# # print(shap_values[0], len(shap_values[0]))
# prompt = build_prompt(X.iloc[0].tolist(), config['features']['final'], shap_values, float(prediction[0])) 
# # print('prompt')
# # print(prompt)  
