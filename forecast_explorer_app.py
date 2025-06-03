
import streamlit as st
import pandas as pd
import sqlite3
import altair as alt

# Load data from SQLite
conn = sqlite3.connect("forecast_demo.db")
df = pd.read_sql("SELECT * FROM forecasts", conn)
conn.close()

st.title("🌽 Seed Production Forecast")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filter Options")
    products = st.multiselect("Select Products", df["PRODUCT"].unique(), default=df["PRODUCT"].unique())
    states = st.multiselect("Select States", df["STATE"].unique(), default=df["STATE"].unique())
    years = st.multiselect("Select Years", sorted(df["SALESYEAR"].unique()), default=sorted(df["SALESYEAR"].unique()))

# Filter dataset
filtered = df[df["PRODUCT"].isin(products) & df["STATE"].isin(states) & df["SALESYEAR"].isin(years)]

# Forecast plot
st.subheader("📈 Forecasted Units Sold")
if filtered.empty:
    st.warning("No data available for selected filters.")
else:
    chart = alt.Chart(filtered).mark_line(point=True).encode(
        x="SALESYEAR:O",
        y="Units_Pred:Q",
        color="PRODUCT:N",
        tooltip=["PRODUCT", "STATE", "SALESYEAR", "Units_Pred", "Lower_Bound", "Upper_Bound"]
    ).properties(width=700, height=400)

    ci_area = alt.Chart(filtered).mark_area(opacity=0.2).encode(
        x="SALESYEAR:O",
        y="Lower_Bound:Q",
        y2="Upper_Bound:Q",
        color="PRODUCT:N"
    )

    st.altair_chart(ci_area + chart, use_container_width=True)

# Table view
st.subheader("🧾 Forecast Table")
st.dataframe(filtered[[
    "PRODUCT", "STATE", "SALESYEAR", "Units_Pred", "Lower_Bound", "Upper_Bound", "LIFECYCLE", "Drivers"
]])

# Download button
csv = filtered.to_csv(index=False)
st.download_button("📥 Download Forecast as CSV", csv, file_name="filtered_forecast.csv", mime="text/csv")
