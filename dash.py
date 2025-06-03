import streamlit as st
import pandas as pd
import plotly.express as px

# 页面配置
st.set_page_config(page_title="Burkina Faso Seed Dashboard", layout="wide")

# 标题与标签
st.title("🌾 Country Dashboard: Burkina Faso")
st.subheader("B1. Number of active seed companies/producers")

# 模拟数据
data = {
    'Crop': ['Cowpea', 'Cowpea', 'Cowpea', 'Maize', 'Maize', 'Maize', 'Rice', 'Rice', 'Rice', 'Sorghum', 'Sorghum', 'Sorghum'],
    'Year': [2018, 2020, 2022] * 4,
    'Number of Companies': [16, 17, 17, 17, 21, 19, 16, 18, 17, 14, 16, 13]
}
df = pd.DataFrame(data)

# 选择器
selected_crops = st.multiselect("Select crop(s):", options=df["Crop"].unique(), default=df["Crop"].unique())
selected_years = st.multiselect("Select year(s):", options=sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))

# 过滤数据
filtered_df = df[(df["Crop"].isin(selected_crops)) & (df["Year"].isin(selected_years))]

# 条形图
fig = px.bar(
    filtered_df,
    x="Crop",
    y="Number of Companies",
    color="Year",
    barmode="group",
    labels={"Number of Companies": "Number of companies/producers"},
    title="Number of active seed companies/producers per crop and year"
)
fig.update_layout(xaxis_title="Crop", yaxis_title="Number of companies/producers", legend_title="Year")

# 显示图表
st.plotly_chart(fig, use_container_width=True)

# 数据来源
st.markdown("**Source:** SODA")
