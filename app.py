import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Marketing Revenue Prediction", layout="wide")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.metric-card {
    background-color: #111827;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    color: white;
}
.section-box {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 10px;
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
st.title("🚀 Marketing Revenue Prediction System")

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader("📊 Key Business Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='metric-card'>Ad Spend<br><h3>{df['ad_spend'].mean():.0f}</h3></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'>Revenue<br><h3>{df['sales_revenue'].mean():.0f}</h3></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'>CTR<br><h3>{df['click_through_rate'].mean():.3f}</h3></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card'>CLV<br><h3>{df['customer_lifetime_value'].mean():.0f}</h3></div>", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Business Problem", "Solution", "EDA", "Heatmap", "Prediction"]
)

# -----------------------------
# BUSINESS PROBLEM
# -----------------------------
if page == "Business Problem":
    st.subheader("📌 Business Problem")

    st.markdown("""
    <div class='section-box'>
    Businesses struggle to understand how marketing spend, pricing strategies, 
    and customer behavior impact sales revenue.

    👉 This leads to poor decision-making and reduced profitability.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# SOLUTION
# -----------------------------
elif page == "Solution":
    st.subheader("💡 Solution Approach")

    st.markdown("""
    <div class='section-box'>
    A machine learning solution using <b>KNN Regression</b> is built to:
    - Analyze key factors affecting revenue  
    - Predict revenue in real-time  
    - Support data-driven decisions  
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# EDA (UPDATED)
# -----------------------------
elif page == "EDA":

    st.subheader("📊 Exploratory Data Analysis")

    analysis_type = st.radio("Select Analysis Type", ["Univariate", "Bivariate"])

    numeric_cols = df.select_dtypes(include=np.number).columns

    # -----------------------------
    # UNIVARIATE
    # -----------------------------
    if analysis_type == "Univariate":
        st.markdown("### 🔹 Univariate Analysis")

        col = st.selectbox("Select Feature", numeric_cols)

        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(df[col], bins=30)
        ax.set_title(f"Distribution of {col}")

        st.pyplot(fig)

    # -----------------------------
    # BIVARIATE
    # -----------------------------
    else:
        st.markdown("### 🔹 Bivariate Analysis")

        x = st.selectbox("X-axis", numeric_cols)
        y = st.selectbox("Y-axis", numeric_cols)

        fig, ax = plt.subplots(figsize=(5,3))
        ax.scatter(df[x], df[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)

        st.pyplot(fig)

# -----------------------------
# HEATMAP (IMPROVED SIZE)
# -----------------------------
elif page == "Heatmap":
    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10,7))  # 🔥 Increased size
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        annot_kws={"size": 9}  # 🔥 Bigger text
    )

    st.pyplot(fig)

# -----------------------------
# PREDICTION
# -----------------------------
elif page == "Prediction":

    st.subheader("🎯 Predict Revenue")

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

    if st.button("🚀 Predict Revenue"):

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

        # Encoding
        segment_map = {"Low": 0, "Medium": 1, "High": 2}
        input_df["customer_segment"] = input_df["customer_segment"].map(segment_map)

        # Match features
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale
        features_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(features_scaled)[0]

        st.markdown(
            f"""
            <div style="background-color:#16a34a;padding:20px;border-radius:10px;color:white;text-align:center;">
                <h2>💰 Predicted Revenue</h2>
                <h1>₹ {prediction:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )