# 🚀 Marketing Revenue Prediction System (KNN-Based)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/Machine%20Learning-KNN-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

---

## 📌 Overview

This project is a **Machine Learning-powered interactive dashboard** that predicts marketing revenue based on campaign performance and customer behavior.

It helps businesses answer:

👉 *“How much revenue can we expect from a marketing strategy?”*

Built using **K-Nearest Neighbors (KNN)** and deployed with **Streamlit**, the app provides real-time predictions and visual insights.

---

## ✨ Features

* 🎯 **Revenue Prediction using KNN**
* 📊 Interactive dashboard with Streamlit
* 📈 Univariate & Bivariate analysis
* 🔥 Correlation Heatmap with values
* ⚡ Real-time predictions based on user input
* 🎨 Premium UI (clean & compact)

---

## 🧠 Machine Learning Workflow

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

```bash
marketing-revenue-prediction/
│── app.py
│── requirements.txt
│── runtime.txt
│── Datasets/
│   ├── train.csv        # Training dataset
│   └── test.csv         # Testing dataset
│── notebooks/
│   ├── eda.ipynb
│   └── models/
│       ├── knn_model.pkl
│       ├── scaler.pkl
│       ├── features.pkl
│       ├── encoders.pkl
```

---

## 📊 Dataset Description

* **train.csv** → Used for training the model
* **test.csv** → Used for evaluation

### Features include:

* Ad Spend
* Market Reach
* Impressions
* CTR (Click Through Rate)
* Price & Discount Rate
* Competition Index
* Seasonality Index
* Customer Segment
* Customer Lifetime Value

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Siva-pa/marketing-revenue-prediction.git
cd marketing-revenue-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

👉 https://marketing-revenue-prediction.streamlit.app/

---

## 📸 Screenshots

### 🔹 Dashboard

<img width="1919" height="719" alt="image" src="https://github.com/user-attachments/assets/24429b4e-80c0-4a51-8f1d-3d80844c0e49" />


### 🔹 Prediction Output

<img width="1919" height="871" alt="image" src="https://github.com/user-attachments/assets/1812c4bb-4857-4a3a-8804-3c2f2b001a41" />



---

## 🧪 Example Use Case

| Scenario             | Input                | Prediction   |
| -------------------- | -------------------- | ------------ |
| Low Campaign Budget  | Low spend, low CTR   | Low revenue  |
| High Campaign Budget | High spend, high CTR | High revenue |

---

## 🚀 Key Learnings

* Handling **feature mismatch during deployment**
* Importance of **consistent preprocessing pipelines**
* Real-world debugging of ML systems
* Building end-to-end ML applications with Streamlit

---

## 🔮 Future Enhancements

* 📈 Model explainability (SHAP / feature importance)
* 🤖 Compare multiple ML models
* 🌐 Deploy with custom domain
* 📊 Advanced business insights dashboard

---

## 👨‍💻 Author

**Siva Kishore Pasupuleti**

* 🔗 GitHub: https://github.com/Siva-pa

---

## ⭐ Support

If you like this project:

👉 Give it a ⭐ on GitHub
👉 Share it with others

---
