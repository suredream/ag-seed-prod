import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Tabbed Menu Example", layout="wide")

# 页面标题
st.title("Seed Systems Dashboard")

# 标签页菜单
selected = option_menu(
    menu_title=None,  # 不显示主标题
    options=[
        "Country Overview",
        "Research and Development",
        "Industry Competitiveness",
        "Seed Policy and Regulations",
        "Institutional Support",
        "Service to Smallholder Farmers",
    ],
    icons=["globe", "flask", "industry", "file-earmark-text", "building", "people"],
    menu_icon="cast",
    default_index=5,  # 默认选中最后一个
    orientation="horizontal",
)

# 内容区
st.write(f"### Selected Tab: {selected}")
st.info(f"Content for: **{selected}** will be displayed here.")
