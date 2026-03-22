import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Marketing Intelligence Dashboard", layout="wide")

# -----------------------------
# CUSTOM CSS (COMPACT + PREMIUM)
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.metric-card {
    background-color: #111827;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("Datasets/train.csv")

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
model = pickle.load(open("notebooks/models/knn_model.pkl", "rb"))
scaler = pickle.load(open("notebooks/models/scaler.pkl", "rb"))
feature_names = pickle.load(open("notebooks/models/features.pkl", "rb"))

# -----------------------------
# TITLE
# -----------------------------
st.title("🚀 Marketing Intelligence Dashboard")

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("📊 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='metric-card'>Ad Spend<br><h3>{df['ad_spend'].mean():.0f}</h3></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'>Revenue<br><h3>{df['sales_revenue'].mean():.0f}</h3></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'>CTR<br><h3>{df['click_through_rate'].mean():.3f}</h3></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card'>CLV<br><h3>{df['customer_lifetime_value'].mean():.0f}</h3></div>", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Univariate", "Bivariate", "Heatmap", "Prediction"]
)

# -----------------------------
# OVERVIEW
# -----------------------------
if page == "Overview":
    st.subheader("📌 Business Overview")
    st.write("This dashboard predicts revenue using KNN based on marketing data.")

# -----------------------------
# UNIVARIATE
# -----------------------------
elif page == "Univariate":
    st.subheader("📊 Feature Distribution")

    col = st.selectbox("Select Feature", df.select_dtypes(include=np.number).columns)

    fig, ax = plt.subplots(figsize=(4,2))
    ax.hist(df[col], bins=30)
    ax.set_title(col)

    st.pyplot(fig)

# -----------------------------
# BIVARIATE
# -----------------------------
elif page == "Bivariate":
    st.subheader("📈 Feature Relationships")

    num_cols = df.select_dtypes(include=np.number).columns

    x = st.selectbox("X-axis", num_cols)
    y = st.selectbox("Y-axis", num_cols)

    fig, ax = plt.subplots(figsize=(4,2))
    ax.scatter(df[x], df[y])
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    st.pyplot(fig)

# -----------------------------
# HEATMAP (FIXED)
# -----------------------------
elif page == "Heatmap":
    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)

# -----------------------------
# PREDICTION (FULL FIX)
# -----------------------------
elif page == "Prediction":

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

    # Encode categorical
    segment_map = {"Low": 0, "Medium": 1, "High": 2}
    customer_segment = segment_map[customer_segment]

    if st.button("🚀 Predict Revenue"):

        # 🔥 Create input dictionary (same as training columns)
        input_dict = {
            "ad_spend": ad_spend,
            "market_reach": market_reach,
            "impressions": impressions,
            "click_through_rate": ctr,
            "price": price,
            "discount_rate": discount,
            "competition_index": competition,
            "seasonality_index": seasonality,
            "customer_segment": customer_segment,
            "customer_lifetime_value": clv
        }

        input_df = pd.DataFrame([input_dict])

        # 🔥 CRITICAL FIX → match training feature count (handles 16 features)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Debug (optional)
        # st.write("Input DF:", input_df)

        # Scale
        features_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(features_scaled)[0]

        # PREMIUM OUTPUT CARD
        st.markdown(
            f"""
            <div style="background-color:#16a34a;padding:20px;border-radius:10px;color:white;text-align:center;">
                <h2>💰 Predicted Revenue</h2>
                <h1>₹ {prediction:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )