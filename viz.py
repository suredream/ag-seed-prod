import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler

# 设置页面配置
st.set_page_config(
    page_title="XGBoost模型分析仪表板",
    page_icon="📊",
    layout="wide"
)

# 标题和描述
st.title("🚀 XGBoost模型可视化分析仪表板")
st.markdown("---")

# 侧边栏配置
st.sidebar.title("⚙️ 模型配置")

# 选择数据集
dataset_choice = st.sidebar.selectbox(
    "选择数据集",
    ["Seed Production", "模拟回归数据"]
)

# 模型参数配置
st.sidebar.subheader("XGBoost参数")
n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100)
max_depth = st.sidebar.slider("max_depth", 3, 10, 6)
learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1)

@st.cache_data
def load_data(choice):
    """加载数据集"""
    df = pd.read_csv('case_study_data.csv')
    df['idx'] = df.index  # Save original index
    df['Years_Since_Release'] = df['SALESYEAR'] - df['RELEASE_YEAR']

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['STATE', 'LIFECYCLE'])
    ordinal_cols = ['DROUGHT_TOLERANCE', 'BRITTLE_STALK', 'PLANT_HEIGHT', 'RELATIVE_MATURITY']
    scaler = MinMaxScaler()
    df_encoded[ordinal_cols] = scaler.fit_transform(df_encoded[ordinal_cols])
    df_encoded = df_encoded.drop(columns=['PRODUCT', 'RELEASE_YEAR'])
    if 'Lifecycle Stage_Phaseout' in df_encoded.columns:
        df_encoded = df_encoded[df_encoded['Lifecycle Stage_Phaseout'] == 0]
        df_encoded = df_encoded.drop(columns=['Lifecycle Stage_Phaseout'])
    X = df_encoded.drop(columns=['UNITS','idx'])
    # print(X.dtypes) # bool cann't be used in TreeExplainer compute
    y = df_encoded['UNITS']

    # if choice == "波士顿房价":
    #     # 使用sklearn的make_regression创建类似的数据集
    #     X, y = make_regression(n_samples=506, n_features=13, noise=10, random_state=42)
    #     feature_names = [f'feature_{i}' for i in range(13)]
    #     X = pd.DataFrame(X, columns=feature_names)
    #     y = pd.Series(y, name='target')
    # else:
    #     X, y = make_regression(n_samples=1000, n_features=10, noise=5, random_state=42)
    #     feature_names = [f'feature_{i}' for i in range(10)]
    #     X = pd.DataFrame(X, columns=feature_names)
    #     y = pd.Series(y, name='target')
    
    return X, y

@st.cache_data
def train_model(X, y, n_est, max_d, lr):
    """训练XGBoost模型"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        learning_rate=lr,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test

def calculate_confidence_intervals(model, X_test, y_test, confidence=0.95):
    """计算置信区间（使用自助法估计）"""
    predictions = []
    n_bootstrap = 100
    
    # 获取训练数据的索引
    n_samples = len(X_test)
    
    for _ in range(n_bootstrap):
        # 自助采样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X_test.iloc[indices]
        
        # 预测
        pred = model.predict(X_bootstrap)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 计算置信区间
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    lower_bound = np.percentile(predictions, lower_percentile, axis=0)
    upper_bound = np.percentile(predictions, upper_percentile, axis=0)
    
    return lower_bound, upper_bound

# 加载数据和训练模型
X, y = load_data(dataset_choice)
model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_model(
    X, y, n_estimators, max_depth, learning_rate
)

# 主要内容区域
col1, col2 = st.columns(2)

# 1. 评估指标
with col1:
    st.subheader("📈 模型评估指标")
    
    # 计算指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # 创建指标表格
    metrics_df = pd.DataFrame({
        '指标': ['RMSE', 'R²', 'MAE'],
        '训练集': [f'{train_rmse:.3f}', f'{train_r2:.3f}', f'{train_mae:.3f}'],
        '测试集': [f'{test_rmse:.3f}', f'{test_r2:.3f}', f'{test_mae:.3f}']
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # 预测vs实际值散点图
    fig_scatter = go.Figure()
    
    fig_scatter.add_trace(go.Scatter(
        x=y_test, y=y_pred_test,
        mode='markers',
        name='测试集',
        marker=dict(color='blue', opacity=0.6)
    ))
    
    # 添加完美预测线
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='完美预测',
        line=dict(color='red', dash='dash')
    ))
    
    fig_scatter.update_layout(
        title='预测值 vs 实际值',
        xaxis_title='实际值',
        yaxis_title='预测值',
        height=400
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# 2. 特征重要性
with col2:
    st.subheader("🎯 特征重要性")
    
    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    # 创建水平条形图
    fig_importance = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='特征重要性排序',
        height=400
    )
    
    fig_importance.update_layout(
        xaxis_title='重要性分数',
        yaxis_title='特征'
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)

# 3. 置信区间分析
st.subheader("📊 预测置信区间分析")

col3, col4 = st.columns(2)

with col3:
    confidence_level = st.slider("置信水平", 0.8, 0.99, 0.95, 0.01)

with col4:
    n_samples_to_show = st.slider("显示样本数", 10, 50, 20)

# 计算置信区间
lower_bound, upper_bound = calculate_confidence_intervals(model, X_test, y_test, confidence_level)

# 选择要显示的样本
indices_to_show = np.random.choice(len(y_test), n_samples_to_show, replace=False)
indices_to_show = sorted(indices_to_show)

# 创建置信区间图
fig_ci = go.Figure()

# 添加置信区间
fig_ci.add_trace(go.Scatter(
    x=list(range(len(indices_to_show))),
    y=upper_bound[indices_to_show],
    fill=None,
    mode='lines',
    line_color='rgba(0,100,80,0)',
    showlegend=False
))

fig_ci.add_trace(go.Scatter(
    x=list(range(len(indices_to_show))),
    y=lower_bound[indices_to_show],
    fill='tonexty',
    mode='lines',
    line_color='rgba(0,100,80,0)',
    name=f'{confidence_level*100:.0f}% 置信区间',
    fillcolor='rgba(0,100,80,0.2)'
))

# 添加预测值
fig_ci.add_trace(go.Scatter(
    x=list(range(len(indices_to_show))),
    y=y_pred_test[indices_to_show],
    mode='markers+lines',
    name='预测值',
    line=dict(color='blue')
))

# 添加实际值
fig_ci.add_trace(go.Scatter(
    x=list(range(len(indices_to_show))),
    y=y_test.iloc[indices_to_show],
    mode='markers',
    name='实际值',
    marker=dict(color='red', size=8)
))

fig_ci.update_layout(
    title=f'预测置信区间 (置信水平: {confidence_level*100:.0f}%)',
    xaxis_title='样本索引',
    yaxis_title='值',
    height=500
)

st.plotly_chart(fig_ci, use_container_width=True)

# 4. SHAP分析
st.subheader("🔍 SHAP可解释性分析")

# 计算SHAP值
@st.cache_data
def calculate_shap_values(_model, _X_train, _X_test):
    explainer = shap.TreeExplainer(_model)
    shap_values_train = explainer.shap_values(_X_train)
    shap_values_test = explainer.shap_values(_X_test)
    return explainer, shap_values_train, shap_values_test

try:
    explainer, shap_values_train, shap_values_test = calculate_shap_values(model, X_train, X_test)
    shap_success = True
except Exception as e:
    st.error(f"SHAP计算失败: {str(e)}")
    st.info("将使用模型内置的特征重要性作为替代")
    shap_success = False

if shap_success:

    # SHAP汇总图
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("SHAP汇总图")
        
        fig_shap_summary, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_test, X_test, show=False, max_display=10)
        st.pyplot(fig_shap_summary, bbox_inches='tight')
        plt.close()

    with col6:
        st.subheader("SHAP特征重要性")
        
        fig_shap_bar, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False, max_display=10)
        st.pyplot(fig_shap_bar, bbox_inches='tight')
        plt.close()

    # 5. 单个预测的SHAP解释
    st.subheader("🎯 单个预测的SHAP解释")

    sample_idx = st.selectbox(
        "选择要分析的样本索引",
        range(min(20, len(X_test)))
    )

    col7, col8 = st.columns(2)

    with col7:
        # 显示样本信息
        st.write("**样本特征值:**")
        sample_features = X_test.iloc[sample_idx]
        feature_df = pd.DataFrame({
            '特征': sample_features.index,
            '值': sample_features.values
        })
        st.dataframe(feature_df)
        
        # 预测结果
        prediction = y_pred_test[sample_idx]
        actual = y_test.iloc[sample_idx]
        st.write(f"**预测值:** {prediction:.3f}")
        st.write(f"**实际值:** {actual:.3f}")
        st.write(f"**误差:** {abs(prediction - actual):.3f}")

    with col8:
        # SHAP瀑布图
        st.write("**SHAP解释 (瀑布图):**")
        
        try:
            # 尝试使用新版本SHAP的waterfall_plot
            fig_waterfall, ax = plt.subplots(figsize=(10, 8))
            
            # 创建Explanation对象 (适用于新版SHAP)
            if hasattr(shap, 'Explanation'):
                explanation = shap.Explanation(
                    values=shap_values_test[sample_idx],
                    base_values=explainer.expected_value,
                    data=X_test.iloc[sample_idx].values,
                    feature_names=X_test.columns.tolist()
                )
                shap.waterfall_plot(explanation, show=False)
            else:
                # 降级到旧版本API
                shap.waterfall_plot(
                    explainer.expected_value, 
                    shap_values_test[sample_idx], 
                    X_test.iloc[sample_idx]
                )
            
            st.pyplot(fig_waterfall, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            # 如果瀑布图失败，使用力图作为替代
            st.write("**SHAP解释 (力图):**")
            fig_force, ax = plt.subplots(figsize=(12, 6))
            
            # 创建force plot
            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values_test[sample_idx],
                X_test.iloc[sample_idx],
                matplotlib=True,
                show=False
            )
            
            st.pyplot(fig_force, bbox_inches='tight')
            plt.close()
            
            # 显示错误信息（可选）
            with st.expander("调试信息"):
                st.write(f"瀑布图错误: {str(e)}")
                st.write("已使用力图作为替代可视化")

    # 6. 特征驱动因子分析
    st.subheader("📋 特征驱动因子分析")

    # 计算每个特征的平均SHAP值贡献
    feature_contributions = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap_abs': np.abs(shap_values_test).mean(0),
        'mean_shap': shap_values_test.mean(0)
    })

    feature_contributions = feature_contributions.sort_values('mean_shap_abs', ascending=False)

    # 创建驱动因子图
    fig_drivers = px.bar(
        feature_contributions.head(10),
        x='mean_shap',
        y='feature',
        orientation='h',
        title='Top 10 特征驱动因子 (平均SHAP值)',
        color='mean_shap',
        color_continuous_scale='RdBu_r'
    )

    fig_drivers.update_layout(height=500)
    st.plotly_chart(fig_drivers, use_container_width=True)

    # 7. 模型性能总结
    st.subheader("📝 模型性能总结")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric("测试集R²", f"{test_r2:.3f}")
        st.metric("测试集RMSE", f"{test_rmse:.3f}")

    with summary_col2:
        st.metric("最重要特征", feature_contributions.iloc[0]['feature'])
        st.metric("特征数量", len(X.columns))

    with summary_col3:
        st.metric("训练样本数", len(X_train))
        st.metric("测试样本数", len(X_test))

    # 8. 数据下载
    st.subheader("💾 结果下载")

    col9, col10 = st.columns(2)

    with col9:
        # 准备预测结果数据
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred_test,
            'lower_bound': lower_bound[:len(y_test)],
            'upper_bound': upper_bound[:len(y_test)]
        })
        
        csv_results = results_df.to_csv(index=False)
        st.download_button(
            "下载预测结果",
            csv_results,
            "predictions.csv",
            "text/csv"
        )

    with col10:
        if shap_success:
            # 准备SHAP值数据
            shap_df = pd.DataFrame(shap_values_test, columns=X_test.columns)
            csv_shap = shap_df.to_csv(index=False)
            
            st.download_button(
                "下载SHAP值",
                csv_shap,
                "shap_values.csv",
                "text/csv"
            )
        else:
            # 如果SHAP失败，提供特征重要性数据
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            })
            csv_importance = importance_df.to_csv(index=False)
            
            st.download_button(
                "下载特征重要性",
                csv_importance,
                "feature_importance.csv",
                "text/csv"
            )

    # 页脚
    st.markdown("---")
    st.markdown("*此应用展示了XGBoost模型的全面分析，包括性能评估、特征重要性、置信区间和可解释性分析。*")