# 🚕 NYC Taxi Trip Duration Prediction

## 📌 Project Overview

This project applies a full data science pipeline on NYC Taxi dataset to predict trip duration using machine learning.

---

## 🔄 Pipeline Steps

### 1. Data Collection

* Dataset: NYC Taxi Trips
* Source: Kaggle
* Target: `trip_duration`

---

### 2. Data Cleaning

* Removed duplicates
* Handled missing values
* Converted datetime

---

### 3. Feature Engineering

* Extracted:

  * hour
  * weekday
* Calculated:

  * distance (Haversine)

---

### 4. Feature Selection

* SelectKBest
* Random Forest Importance

---

### 5. Modeling

* Model: Random Forest Regressor
* Train/Test Split: 80/20

---

### 6. Evaluation

* Metrics:

  * MAE
  * RMSE
  * R²

---

## 🚀 Streamlit App

### Features:

* Input trip data (Uber-style)
* Predict trip duration
* Estimate trip price
* Interactive map (Mapbox)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## 📊 Example Output

* Distance
* Duration (minutes)
* Price estimate

---

## 🧠 Tech Stack

* Python
* Pandas
* Scikit-learn
* Streamlit
* Plotly / Mapbox
