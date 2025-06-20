import pandas as pd
import toml
import os
import argparse
from src.pipelines.xgb import train_and_save, inference 
from src.pipelines.residue import train_and_save_residual, inference_residual
from src.pipelines.explain import call_explain, build_prompt
import joblib

# === parser.add_argument ===
parser = argparse.ArgumentParser(description='Train and predict model.')
# parser.add_argument('--model', type=str, default='residual', choices=['residual', 'xgb'], help='Specify the model type (residual or xgb).')
parser.add_argument('--model', type=str, default='xgb', choices=['residual', 'xgb'], help='Specify the model type (residual or xgb).')
parser.add_argument('--pred_only', default=False, action='store_false', help='Only perform prediction, skip training.')
args = parser.parse_args()
# print(args)

# === load config ===
model_type = args.model
if model_type == 'residual':
    config = toml.load("config/residual.toml")
    model_dict = {'train_and_save': train_and_save_residual, 'inference': inference_residual}
elif model_type == 'xgb':
    config = toml.load("config/xgb.toml")
    model_dict = {'train_and_save': train_and_save, 'inference': inference}
else:
    raise ValueError("Invalid model type. Choose 'residual' or 'xgb'.")

# === train ===
if not args.pred_only:
# if not os.path.exists(config['output']['model_path']):
    data_path = config['input']['data_path']
    df = pd.read_csv(data_path)
    model_dict['train_and_save'](df, config)
    print(f"✅ {model_type} training done")


# # === Inference from a dict ===
# input_dict = {
#     'PRODUCT': 'P123',
#     'SALESYEAR': 2024,
#     'RELEASE_YEAR': 2020,
#     'DISEASE_RESISTANCE': 3,
#     'INSECT_RESISTANCE': 4,
#     'PROTECTION': 2,
#     'DROUGHT_TOLERANCE': 5.0,
#     'BRITTLE_STALK': None,        # 可缺失
#     'PLANT_HEIGHT': 6.0,
#     'RELATIVE_MATURITY': 3,
#     'STATE': 'Texas',
#     'LIFECYCLE': 'EXPANSION'
# }

# base_model = joblib.load(config['output']['base_model_path'])

# input_df = pd.DataFrame([input_dict])
# pred_dict = model_dict['inference'](input_df, input_dict['PRODUCT'], config)
# explain = pred_dict['explain']
# prompt = build_prompt(explain['input'], explain['feature_names'], explain['shap_values'], explain['pred'])
# print(call_explain(prompt))