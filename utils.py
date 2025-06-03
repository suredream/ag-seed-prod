import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data():
    """加载数据集并返回 features, target, 以及用于切分的原始分组信息"""
    df = pd.read_csv('case_study_data.csv')
    df['idx'] = df.index  # Save original index
    df['Years_Since_Release'] = df['SALESYEAR'] - df['RELEASE_YEAR']

    state = df['STATE'].copy()
    lifecycle = df['LIFECYCLE'].copy()
    release_years = df['Years_Since_Release'].copy()

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['STATE', 'LIFECYCLE'])
    ordinal_cols = ['DROUGHT_TOLERANCE', 'BRITTLE_STALK', 'PLANT_HEIGHT', 'RELATIVE_MATURITY']
    scaler = MinMaxScaler()
    df_encoded[ordinal_cols] = scaler.fit_transform(df_encoded[ordinal_cols])
    df_encoded = df_encoded.drop(columns=['PRODUCT', 'RELEASE_YEAR'])
    if 'Lifecycle Stage_Phaseout' in df_encoded.columns:
        df_encoded = df_encoded[df_encoded['Lifecycle Stage_Phaseout'] == 0]
        df_encoded = df_encoded.drop(columns=['Lifecycle Stage_Phaseout'])

    X = df_encoded.drop(columns=['UNITS', 'idx'])
    y = df_encoded['UNITS']

    return X, y, state.loc[X.index], lifecycle.loc[X.index], release_years.loc[X.index]


def load_Xy():
    """加载数据集并返回 features, target, 以及用于切分的原始分组信息"""
    df = pd.read_csv('case_study_data.csv')
    df['idx'] = df.index  # Save original index
    df['Years_Since_Release'] = df['SALESYEAR'] - df['RELEASE_YEAR']

    state = df['STATE'].copy()
    lifecycle = df['LIFECYCLE'].copy()
    release_years = df['Years_Since_Release'].copy()

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['STATE', 'LIFECYCLE'])
    ordinal_cols = ['DROUGHT_TOLERANCE', 'BRITTLE_STALK', 'PLANT_HEIGHT', 'RELATIVE_MATURITY']
    scaler = MinMaxScaler()
    df_encoded[ordinal_cols] = scaler.fit_transform(df_encoded[ordinal_cols])
    df_encoded = df_encoded.drop(columns=['PRODUCT', 'RELEASE_YEAR'])
    if 'Lifecycle Stage_Phaseout' in df_encoded.columns:
        df_encoded = df_encoded[df_encoded['Lifecycle Stage_Phaseout'] == 0]
        df_encoded = df_encoded.drop(columns=['Lifecycle Stage_Phaseout'])

    X = df_encoded.drop(columns=['UNITS', 'idx'])
    y = df_encoded['UNITS']

    return X, y

def compute_drivers(X, model, top_n=3):
    import shap

    # 可选：特征映射字典
    feature_name_map = {
        'DROUGHT_TOLERANCE': '干旱耐受性',
        'PLANT_HEIGHT': '植株高度',
        'BRITTLE_STALK': '茎杆脆性',
        'RELATIVE_MATURITY': '相对成熟度',
        'SALESYEAR': '销售年份',
        'Years_Since_Release': '发布以来年数',
    }

    X_float = X.astype(float)
    explainer = shap.Explainer(model, X_float)
    shap_values = explainer(X_float)

    driver_texts = []

    for i in range(len(X_float)):
        row = shap_values[i]
        shap_contribs = list(zip(X.columns, row.values))
        sample = X.iloc[i]

        # 解释项筛选规则
        def is_valid_feature(feat, value):
            if feat.startswith("STATE_") or feat.startswith("LIFECYCLE_"):
                return sample[feat] == 1  # 只解释当前为 True 的 one-hot 特征
            return True  # 其他数值特征保留

        # 排序和过滤
        top_features = sorted(
            filter(lambda fv: is_valid_feature(fv[0], fv[1]), shap_contribs),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        # 构造解释字符串
        driver_str = ", ".join([
            f"{'↑' if val > 0 else '↓'} " + (
                f"在 {feat.replace('STATE_', '')} 州" if feat.startswith("STATE_") else
                f"{feat.replace('LIFECYCLE_', '').title()} 阶段" if feat.startswith("LIFECYCLE_") else
                feature_name_map.get(feat, feat.replace('_', ' ').title())
            )
            for feat, val in top_features
        ])

        driver_texts.append(driver_str)

    return driver_texts
