import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error


import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

class DistributionComparator:
    def __init__(self, feature_types=None):
        """
        feature_types: dict，指定每个特征是 'numerical' 还是 'categorical'，如果不提供将自动推断
        """
        self.feature_types = feature_types or {}

    def infer_feature_type(self, series):
        if self.feature_types:
            return self.feature_types.get(series.name, 'numerical')
        return 'categorical' if series.nunique() < 20 or series.dtype == 'object' else 'numerical'

    def compare(self, X_train, X_test, top_k=10, show_plot=False):
        results = []
        for col in X_train.columns:
            feature_type = self.infer_feature_type(X_train[col])
            if feature_type == 'numerical':
                stat, p_value = ks_2samp(X_train[col], X_test[col])
                results.append({'Feature': col, 'Type': 'Numerical', 'Stat': stat, 'P-Value': p_value})
                if show_plot:
                    self._plot_numerical(X_train[col], X_test[col], col)
            else:
                contingency = pd.crosstab(X_train[col], np.ones(len(X_train)))
                contingency_test = pd.crosstab(X_test[col], np.zeros(len(X_test)))
                combined = contingency.add(contingency_test, fill_value=0)
                chi2, p_value, _, _ = chi2_contingency(combined)
                results.append({'Feature': col, 'Type': 'Categorical', 'Stat': chi2, 'P-Value': p_value})
                if show_plot:
                    self._plot_categorical(X_train[col], X_test[col], col)
        df = pd.DataFrame(results).sort_values("P-Value")
        return df.head(top_k)

    def _plot_numerical(self, train_col, test_col, col_name):
        plt.figure(figsize=(6, 4))
        sns.kdeplot(train_col, label='Train', fill=True)
        sns.kdeplot(test_col, label='Test', fill=True)
        plt.title(f'Distribution of {col_name}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_categorical(self, train_col, test_col, col_name):
        train_freq = train_col.value_counts(normalize=True)
        test_freq = test_col.value_counts(normalize=True)
        df_plot = pd.DataFrame({'Train': train_freq, 'Test': test_freq}).fillna(0)
        df_plot.plot(kind='bar', figsize=(6, 4), title=f'Distribution of {col_name}')
        plt.tight_layout()
        plt.show()


from sklearn.base import clone

class GeneralizationCV:
    def __init__(self, model, metrics=None, n_splits=5, top_k_drift=5):
        self.model = model
        self.metrics = metrics or {
            "MAE": mean_absolute_error,
            "RMSE": lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred),
            "R2": r2_score
        }
        self.n_splits = n_splits
        self.top_k_drift = top_k_drift
        self.results = []
        self.drift_reports = []  # ✅ 每fold的特征偏移记录

    def _evaluate(self, y_true, y_pred):
        return {name: func(y_true, y_pred) for name, func in self.metrics.items()}

    def _log_result(self, strategy, train_group, test_group, scores, drift_df):
        result = {"Strategy": strategy, "Train Group": train_group, "Test Group": test_group}
        result.update(scores)
        self.results.append(result)
        self.drift_reports.append({
            "Strategy": strategy,
            "Train Group": train_group,
            "Test Group": test_group,
            "Feature Drift": drift_df
        })

    def _run_fold(self, X, y, train_idx, test_idx, strategy, train_group_name, test_group_name):
        model = clone(self.model)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores = self._evaluate(y_test, y_pred)

        # ✅ 特征分布差异分析
        comparator = DistributionComparator()
        drift_df = comparator.compare(X_train, X_test, top_k=self.top_k_drift, show_plot=False)
        self._log_result(strategy, train_group_name, test_group_name, scores, drift_df)

    def state_cv(self, X, y, groups):
        gkf = GroupKFold(n_splits=self.n_splits)
        for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            self._run_fold(X, y, train_idx, test_idx, "STATE_CV", f"fold_{i}_train", f"fold_{i}_test")

    def lifecycle_cv(self, X, y, groups):
        gkf = GroupKFold(n_splits=4)
        for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            self._run_fold(X, y, train_idx, test_idx, "LIFECYCLE_CV", f"fold_{i}_train", f"fold_{i}_test")

    def temporal_cv(self, X, y, time_col):
        df = X.copy()
        df["target"] = y
        df = df.sort_values(time_col)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for i, (train_idx, test_idx) in enumerate(tscv.split(df)):
            self._run_fold(
                df.drop(columns="target"), df["target"],
                train_idx, test_idx,
                "TEMPORAL_CV", f"split_{i}_train", f"split_{i}_test"
            )

    def get_report(self):
        return pd.DataFrame(self.results)

    def get_feature_drift_report(self):
        # 每一折返回的特征漂移记录列表
        return self.drift_reports

    def reset(self):
        self.results = []
        self.drift_reports = []


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

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


# # 加载数据
# X, y, state_group, lifecycle_group, release_years = load_data()

# # 初始化模型与评估器
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# cv_runner = GeneralizationCV(model)

# # 执行交叉验证策略
# cv_runner.state_cv(X, y, groups=state_group)
# cv_runner.lifecycle_cv(X, y, groups=lifecycle_group)
# cv_runner.temporal_cv(X.assign(Years_Since_Release=release_years), y, time_col="Years_Since_Release")

# # 获取结果
# report_df = cv_runner.get_report()

# # 展示或保存结果
# print(report_df)
from sklearn.ensemble import RandomForestRegressor

X, y, state, lifecycle, release_years = load_data()

cv_runner = GeneralizationCV(RandomForestRegressor(n_estimators=100), top_k_drift=5)

cv_runner.state_cv(X, y, groups=state)
cv_runner.lifecycle_cv(X, y, groups=lifecycle)
cv_runner.temporal_cv(X.assign(Years_Since_Release=release_years), y, time_col="Years_Since_Release")

# 性能报告
performance_report = cv_runner.get_report()
print(performance_report)

# 特征漂移报告（按每个 fold）
drift_reports = cv_runner.get_feature_drift_report()
for drift in drift_reports:
    print(f"\n== {drift['Strategy']} | {drift['Test Group']} ==")
    print(drift['Feature Drift'])
