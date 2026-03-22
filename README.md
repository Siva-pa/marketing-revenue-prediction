# 🚀 Marketing Revenue Prediction System (KNN-Based)

A **Machine Learning-powered Streamlit dashboard** that predicts marketing revenue based on campaign performance, customer behavior, and market conditions.

---

## 📊 Project Overview

Businesses often struggle to understand how marketing efforts translate into revenue.
This project solves that problem by using a **K-Nearest Neighbors (KNN) Regression model** to predict revenue based on key marketing features.

The application is deployed using **Streamlit**, providing an interactive dashboard for real-time predictions and insights.

---

## ✨ Features

* 📈 **Revenue Prediction using KNN**
* 📊 **Interactive Dashboard (Streamlit)**
* 🔥 **Correlation Heatmap with insights**
* 📉 **Univariate & Bivariate Analysis**
* 🎯 **Real-time Input-based Predictions**
* 💡 Clean & Premium UI

---

## 🧠 Machine Learning Pipeline

1. Data Cleaning & Preprocessing
2. Feature Engineering
3. Encoding Categorical Variables
4. Feature Scaling using `StandardScaler`
5. Model Training using **KNeighborsRegressor**
6. Model Saving (`.pkl` files)
7. Deployment with Streamlit

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn (KNN)**
* **Matplotlib, Seaborn**
* **Streamlit**

---

## 📁 Project Structure

```
marketing-revenue-prediction/
│── app.py
│── requirements.txt
│── runtime.txt
│── models/
│   ├── knn_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│── Datasets/
│   └── train.csv
│── notebooks/
│   └── eda.ipynb
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/Siva-pa/marketing-revenue-prediction.git
cd marketing-revenue-prediction
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Run Application

```
streamlit run app.py
```

---

## 🌐 Live Demo

👉 (Add your Streamlit link here)

---

## 📊 Input Features

* Ad Spend
* Market Reach
* Impressions
* Click Through Rate (CTR)
* Price
* Discount Rate
* Competition Index
* Seasonality Index
* Customer Segment
* Customer Lifetime Value

---

## 🎯 Output

* 💰 Predicted Marketing Revenue

---

## 🧪 Example Prediction

| Scenario              | Revenue      |
| --------------------- | ------------ |
| Low Marketing Effort  | Low Revenue  |
| High Marketing Effort | High Revenue |

---

## 🚀 Key Learnings

* Handling **feature mismatch in deployment**
* Importance of **consistent preprocessing**
* Model deployment using **Streamlit**
* Debugging real-world ML issues

---

## 🔥 Future Enhancements

* 📈 Model explainability (SHAP / feature importance)
* 🤖 Compare multiple ML models
* 🌐 Deploy with custom domain
* 📊 Advanced business insights dashboard

---

## 👨‍💻 Author

**Siva Kishore Pasupuleti**

* 🔗 GitHub: https://github.com/Siva-pa
* 💼 Aspiring AI/ML Engineer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
