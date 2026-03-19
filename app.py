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
# Custom CSS (Power BI Style Cards)
# -----------------------------
st.markdown("""
<style>
.card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}
.big-font {
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("Datasets/train.csv")

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("notebooks/models/knn_model.pkl", "rb"))
scaler = pickle.load(open("notebooks/models/scaler.pkl", "rb"))

# -----------------------------
# Title
# -----------------------------
st.title("📊 Marketing Revenue Dashboard")

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("📈 Key Insights")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"<div class='card'><div class='big-font'>Ad Spend</div><br>{df['ad_spend'].mean():.2f}</div>", unsafe_allow_html=True)

with c2:
    st.markdown(f"<div class='card'><div class='big-font'>Revenue</div><br>{df['sales_revenue'].mean():.2f}</div>", unsafe_allow_html=True)

with c3:
    st.markdown(f"<div class='card'><div class='big-font'>CTR</div><br>{df['click_through_rate'].mean():.2f}</div>", unsafe_allow_html=True)

with c4:
    st.markdown(f"<div class='card'><div class='big-font'>CLV</div><br>{df['customer_lifetime_value'].mean():.2f}</div>", unsafe_allow_html=True)

# -----------------------------
# Navigation Buttons
# -----------------------------
st.markdown("### 📂 Sections")

col1, col2, col3, col4, col5, col6 = st.columns(6)

if "page" not in st.session_state:
    st.session_state.page = "home"

with col1:
    if st.button("📌 Problem"):
        st.session_state.page = "problem"

with col2:
    if st.button("🚀 Solution"):
        st.session_state.page = "solution"

with col3:
    if st.button("📊 Univariate"):
        st.session_state.page = "uni"

with col4:
    if st.button("📈 Bivariate"):
        st.session_state.page = "bi"

with col5:
    if st.button("🔥 Heatmap"):
        st.session_state.page = "heatmap"

with col6:
    if st.button("🎯 Prediction"):
        st.session_state.page = "prediction"

# -----------------------------
# Pages
# -----------------------------

# Problem
if st.session_state.page == "problem":
    st.subheader("📌 Problem Statement")
    st.write("""
    Businesses struggle to understand how marketing spend, pricing strategies, and customer behavior impact sales revenue. 
    This project aims to build a predictive system to estimate revenue and support data-driven decisions.
    """)

# Solution
elif st.session_state.page == "solution":
    st.subheader("🚀 Solution")
    st.write("""
    A machine learning solution will be developed by cleaning and preparing the data, analyzing key factors affecting revenue, and training a KNN regression model. 
    The model will be evaluated for accuracy and deployed through a Streamlit application, enabling users to input business parameters and receive real-time sales revenue predictions for better decision-making.
    """)

# Univariate
elif st.session_state.page == "uni":
    st.subheader("📊 Univariate Analysis")

    num_cols = df.select_dtypes(include=np.number).columns
    col = st.selectbox("Select Feature", num_cols)

    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(df[col], bins=30)
    ax.set_title(col)
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
    ax.set_xlabel(x)
    ax.set_ylabel(y)
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

# Prediction
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
        features = np.array([[ad_spend, market_reach, impressions, ctr,
                              price, discount, competition, seasonality,
                              customer_segment, clv]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        st.success(f"💰 Predicted Revenue: ₹ {prediction:,.2f}")
