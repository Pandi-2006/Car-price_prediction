# 🚗 Car Price Prediction Project

**Repository:** [Pandi-2006/Car-price_prediction](https://github.com/Pandi-2006/Car-price_prediction)

---

## 📘 Project Overview

This project predicts the **selling price of used cars** based on their features using **machine learning techniques**.  
It leverages historical car sales data to estimate prices accurately, considering factors like:

- Vehicle age  
- Mileage  
- Engine specifications  
- Fuel type  
- Transmission  
- Seller type  
- Brand and model  

A **Streamlit web app** is also included, allowing users to input car details and get instant price predictions.

---

## 📊 Dataset

- **Source:** Cardekho Used Car Listings Dataset  
- **Rows / Columns:** 15,000+ rows | 14+ columns  

### 🔑 Key Features
| Type | Features |
|------|-----------|
| **Categorical** | `brand`, `model`, `fuel_type`, `transmission_type`, `seller_type` |
| **Numerical** | `vehicle_age`, `km_driven`, `mileage`, `engine`, `max_power`, `seats` |
| **Target Variable** | `selling_price` (in INR) |

---

## 🧹 Data Preprocessing

- Removed **missing values** and **outliers** (top 1% extreme prices/kms)  
- Applied **one-hot encoding** for categorical features  
- Created **new engineered features**:
  - `power_per_engine = max_power / engine`
  - `mileage_engine_factor = mileage * engine`
- Applied **log transformation** on `selling_price` for better model stability  

---

## 🤖 Machine Learning Models

| Model | R² Score | RMSE |
|-------|-----------|------|
| Linear Regression | 0.78 | 0.32 |
| Random Forest | 0.91 | 0.18 |
| **XGBoost** | **0.9345** | **0.1674** |

**✅ Best Model:** XGBoost (used in the Streamlit app)

### 📚 Libraries Used
`scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`, `streamlit`

---

## ⚙️ Steps to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Pandi-2006/Car-price_prediction.git
   cd Car-price_prediction
2. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the streamlit App**
   ```bash
   streamlit run app.py
4. **Use the app**
   - Enter Car detailsbrand, model, mileage, engine, etc.)
   - Click Predict Price
   - View the estimated selling price
  
## 🧾 Project Code Overview

- **data/** → Contains the dataset (`cardekho_dataset.csv`)  
- **models/** → Contains saved model files (`model.pkl`, `feature_names.pkl`)  
- **src/** → Scripts for training and preprocessing  
  - `train_model.py` → Train and evaluate ML models  
  - `data_preprocessing.py` → Preprocess dataset and generate features  
  - `predict.py` → Script to make predictions  
- **app.py** → Streamlit application for live predictions  

---

## 📈 Data Visualization & Insights

### 1️⃣ Selling Price Distribution  
![Selling Price Distribution](assets/selling_price_distribution.png)

### 2️⃣ Actual vs Predicted Prices  
![Actual vs Predicted Prices](assets/actual_vs_predicted.png)
 

---

### 💡 Key Insights

- **Vehicle age** and **kilometers driven** have the strongest **negative impact** on price  
- **Luxury brands** and **newer models** have **higher selling prices**  
- **Engine power** and **mileage** significantly affect car pricing  

---

## 🚀 Future Improvements

- Include additional features like **accident history**, **location**, and **insurance status**  
- Use **ensemble methods** (XGBoost + Random Forest + LightGBM)  
- Deploy a **fully interactive web app** with multiple visualization dashboards  

---

## 📚 References

- [Cardekho Used Car Dataset](https://www.cardekho.com/)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

