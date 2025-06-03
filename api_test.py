import requests
import pandas as pd

# Step 1: 原始输入字典
input_dict = {
    'PRODUCT': 'P123',
    'SALESYEAR': 2024,
    'RELEASE_YEAR': 2020,
    'DISEASE_RESISTANCE': 3,
    'INSECT_RESISTANCE': 4,
    'PROTECTION': 2,
    'DROUGHT_TOLERANCE': 5.0,
    'BRITTLE_STALK': 0,
    'PLANT_HEIGHT': 6.0,
    'RELATIVE_MATURITY': 3,
    'STATE': 'Texas',
    'LIFECYCLE': 'EXPANSION'
}

# Step 2: 转为 DataFrame（用于保留顺序）
input_df = pd.DataFrame([input_dict])

# Step 3: 提取有用的 feature 列（与模型训练时相同顺序）
# ⚠️ 注意：你需要根据实际 feature_cols 保证顺序一致
# feature_cols = [
#     'SALESYEAR', 'RELEASE_YEAR', 'DISEASE_RESISTANCE', 'INSECT_RESISTANCE',
#     'PROTECTION', 'DROUGHT_TOLERANCE', 'BRITTLE_STALK', 'PLANT_HEIGHT',
#     'RELATIVE_MATURITY'
# ]  # example: exclude 'PRODUCT', 'STATE', 'LIFECYCLE' if not numeric features

feature_cols = ['SALESYEAR', 'RELEASE_YEAR',"DISEASE_RESISTANCE", "INSECT_RESISTANCE", "PROTECTION",
                 'DROUGHT_TOLERANCE', 'BRITTLE_STALK', 'PLANT_HEIGHT', 'RELATIVE_MATURITY', 'STATE', 'LIFECYCLE']

# Step 4: 填充缺失值（建议与训练逻辑一致）
# input_df = input_df[feature_cols].fillna(0)

# Step 5: 准备 JSON 载荷
payload = {
    "data": input_dict,
    "confidence": 0.95
}

# Step 6: 调用 FastAPI 预测接口
url = "http://localhost:8000/predict"
response = requests.post(url, json=payload)

# Step 7: 输出结果
if response.status_code == 200:
    result = response.json()
    print("🔮 Predicted Units:", result["prediction"][0])
    # print("🔻 Lower Bound:", result["lower_bound"][0])
    # print("🔺 Upper Bound:", result["upper_bound"][0])
else:
    print("❌ Request failed:", response.status_code)
    print(response.text)
