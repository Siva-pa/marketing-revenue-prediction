import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Marketing Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("Datasets/train.csv")

# -----------------------------
# Load Model + Scaler + Features
# -----------------------------
model = pickle.load(open(r"F:\siva\Hyderabad _ ML\Project\Marketing Revenue Prediction System (KNN-Based)\notebooks\models\knn_model.pkl", "rb"))
scaler = pickle.load(open(r"F:\siva\Hyderabad _ ML\Project\Marketing Revenue Prediction System (KNN-Based)\notebooks\models\scaler.pkl", "rb"))

# IMPORTANT (fix for your error)
feature_names = pickle.load(open(r"F:\siva\Hyderabad _ ML\Project\Marketing Revenue Prediction System (KNN-Based)\notebooks\models\features.pkl", "rb"))

# -----------------------------
# Title
# -----------------------------
st.title("📊 Marketing Revenue Prediction Dashboard")

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("📈 Key Insights")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Avg Ad Spend", f"{df['ad_spend'].mean():.2f}")
c2.metric("Avg Revenue", f"{df['sales_revenue'].mean():.2f}")
c3.metric("Avg CTR", f"{df['click_through_rate'].mean():.2f}")
c4.metric("Avg CLV", f"{df['customer_lifetime_value'].mean():.2f}")

# -----------------------------
# Navigation
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

col1, col2, col3, col4, col5, col6 = st.columns(6)

if col1.button("📌 Problem"):
    st.session_state.page = "problem"
if col2.button("🚀 Solution"):
    st.session_state.page = "solution"
if col3.button("📊 Univariate"):
    st.session_state.page = "uni"
if col4.button("📈 Bivariate"):
    st.session_state.page = "bi"
if col5.button("🔥 Heatmap"):
    st.session_state.page = "heatmap"
if col6.button("🎯 Prediction"):
    st.session_state.page = "prediction"

# -----------------------------
# PAGES
# -----------------------------

# Problem
if st.session_state.page == "problem":
    st.subheader("📌 Problem Statement")
    st.write("""
    Businesses struggle to understand how marketing strategies impact revenue.
    """)

# Solution
elif st.session_state.page == "solution":
    st.subheader("🚀 Solution")
    st.write("""
    We use KNN Regression to predict revenue using marketing and customer data.
    """)

# Univariate
elif st.session_state.page == "uni":
    st.subheader("📊 Univariate Analysis")

    col = st.selectbox("Select Feature", df.select_dtypes(include=np.number).columns)

    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(df[col], bins=30)
    plt.tight_layout()

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.pyplot(fig)

# Bivariate
elif st.session_state.page == "bi":
    st.subheader("📈 Bivariate Analysis")

    num_cols = df.select_dtypes(include=np.number).columns
    x = st.selectbox("X-axis", num_cols)
    y = st.selectbox("Y-axis", num_cols)

    fig, ax = plt.subplots(figsize=(5,3))
    ax.scatter(df[x], df[y])
    plt.tight_layout()

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.pyplot(fig)

# Heatmap
elif st.session_state.page == "heatmap":
    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6,4))
    corr = df.corr(numeric_only=True)

    cax = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)

    fig.colorbar(cax)
    plt.tight_layout()

    st.pyplot(fig)

# -----------------------------
# PREDICTION
# -----------------------------
elif st.session_state.page == "prediction":

    st.subheader("🎯 Revenue Prediction")

    st.sidebar.header("Input Features")

    ad_spend = st.sidebar.number_input("Ad Spend", 0.0)
    market_reach = st.sidebar.number_input("Market Reach", 0.0)
    impressions = st.sidebar.number_input("Impressions", 0.0)
    ctr = st.sidebar.number_input("CTR", 0.0)

    price = st.sidebar.number_input("Price", 0.0)
    discount = st.sidebar.slider("Discount Rate", 0.0, 1.0)

    competition = st.sidebar.number_input("Competition Index", 0.0)
    seasonality = st.sidebar.number_input("Seasonality Index", 0.0)

    customer_segment = st.sidebar.selectbox("Customer Segment", ["Low", "Medium", "High"])
    clv = st.sidebar.number_input("Customer Lifetime Value", 0.0)

    segment_map = {"Low": 0, "Medium": 1, "High": 2}
    customer_segment = segment_map[customer_segment]

    if st.button("🚀 Predict Revenue"):

        # Create input dictionary
        input_dict = {
            "Ad Spend": ad_spend,
            "Market Reach": market_reach,
            "Impressions": impressions,
            "CTR": ctr,
            "Price": price,
            "Discount Rate": discount,
            "Competition Index": competition,
            "Seasonality Index": seasonality,
            "Customer Segment": customer_segment,
            "Customer Lifetime Value": clv
        }

        input_df = pd.DataFrame([input_dict])

        # 🔥 FIX: Match training columns exactly
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale
        features_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(features_scaled)[0]

        st.success(f"💰 Predicted Revenue: ₹ {prediction:,.2f}")

