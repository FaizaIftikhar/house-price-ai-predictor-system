# 🏡 House Price AI Predictor System

An end-to-end **Machine Learning Regression** project that predicts residential house prices based on property characteristics using advanced regression models and feature engineering techniques.

The project demonstrates the complete Machine Learning lifecycle—from data collection and preprocessing to model deployment with **Streamlit Cloud**.

---

## 🌐 Live Demo

🚀 **Try the application here:**

**https://house-price-ai-predictor-system-2eqthcthrc37ksikpmfkjz.streamlit.app/**

No installation required—simply open the link and start predicting house prices.

---

## 📸 Application Preview

<img width="947" height="433" alt="image" src="https://github.com/user-attachments/assets/ae3aff83-bf71-4379-8a2b-a01ef1e21323" />

<img width="950" height="436" alt="image" src="https://github.com/user-attachments/assets/f9f5a803-334b-45b6-a939-d3dd40dcea0f" />

---

# 🚀 Project Overview

House price prediction is one of the most common regression problems in Machine Learning. This project aims to accurately estimate residential property prices based on various housing features using supervised learning algorithms.

The project follows a complete production-ready Machine Learning workflow:

- Data Collection
- Data Understanding & Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Feature Selection
- Train-Test Split
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Model Serialization using Joblib
- Interactive Web Application Development using Streamlit
- Cloud Deployment using Streamlit Community Cloud

---

# 📂 Dataset

This project uses the **House Prices Dataset** from **OpenML**.

Dataset loaded using Scikit-learn:

```python
from sklearn.datasets import fetch_openml

housing = fetch_openml(
    name="house_prices",
    as_frame=True
)

df = housing.frame
