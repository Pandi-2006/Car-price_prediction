<h1>🚗 Car Price Prediction</h1>

<p>This project predicts the <b>selling price of used cars</b> based on their features using <b>machine learning techniques</b>.  
A <b>Streamlit web app</b> is included, allowing users to input car details and instantly get price predictions.</p>

<hr>

<h2>📂 Repository</h2>
<p><a href="https://github.com/Pandi-2006/Car-price_prediction">Car Price Prediction - GitHub</a></p>

<hr>

<h2>📖 Project Overview</h2>
<ul>
  <li>Predicts selling prices of used cars using historical sales data.</li>
  <li>Considers vehicle age, mileage, engine specs, fuel type, transmission, seller type, brand and model.</li>
  <li>Includes an interactive Streamlit app for real-time predictions.</li>
</ul>

<hr>

<h2>📊 Dataset</h2>
<p><b>Source:</b> Cardekho Used Car Dataset<br>
<b>Size:</b> 15,000+ rows, 14+ columns</p>

<h3>Key Features</h3>
<ul>
  <li><b>Categorical:</b> <code>brand</code>, <code>model</code>, <code>fuel_type</code>, <code>transmission_type</code>, <code>seller_type</code></li>
  <li><b>Numerical:</b> <code>vehicle_age</code>, <code>km_driven</code>, <code>mileage</code>, <code>engine</code>, <code>max_power</code>, <code>seats</code></li>
  <li><b>Target:</b> <code>selling_price</code> (INR)</li>
</ul>

<hr>

<h2>⚙️ Preprocessing</h2>
<ul>
  <li>Removed missing values & outliers (top 1% extreme prices/kms).</li>
  <li>One-hot encoding for categorical features.</li>
  <li><b>Feature Engineering:</b>
    <ul>
      <li><code>power_per_engine = max_power / engine</code></li>
      <li><code>mileage_engine_factor = mileage * engine</code></li>
    </ul>
  </li>
  <li>Applied <b>log transformation</b> on <code>selling_price</code> for stability.</li>
</ul>

<hr>

<h2>🤖 Machine Learning Models</h2>
<table>
  <tr><th>Model</th><th>R² Score</th><th>RMSE</th></tr>
  <tr><td>Linear Regression</td><td>0.78</td><td>0.32</td></tr>
  <tr><td>Random Forest</td><td>0.91</td><td>0.18</td></tr>
  <tr><td><b>XGBoost</b></td><td><b>0.9345</b></td><td><b>0.1674</b></td></tr>
</table>

<p>✅ <b>Best Model:</b> XGBoost (used in Streamlit app)</p>

<p><b>Libraries Used:</b> scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, joblib, streamlit</p>

<hr>

<h2>🚀 How to Run the Project</h2>
<pre>
# Clone the repository
git clone https://github.com/Pandi-2006/Car-price_prediction.git
cd Car-price_prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
</pre>
<p>👉 Enter car details (brand, model, mileage, engine, etc.)  
👉 Click <b>Predict Price</b> to get the estimated selling price.</p>

<hr>

<h2>📂 Project Structure</h2>
<pre>
Car-price_prediction/
│
├── data/                 # Dataset (cardekho_dataset.csv)
├── models/               # Saved models (model.pkl, feature_names.pkl)
├── src/                  # Scripts for training & preprocessing
│   ├── train_model.py    # Train & evaluate ML models
│   ├── data_preprocessing.py
│   ├── predict.py        # Make predictions
│
├── app.py                # Streamlit application
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
</pre>

<hr>

<h2>📈 Data Visualization & Insights</h2>

<h3>1. Selling Price Distribution</h3>
<img src="path/to/selling_price_distribution.png" alt="Selling Price Distribution" />

<h3>2. Actual vs Predicted Prices</h3>
<img src="path/to/actual_vs_predicted.png" alt="Actual vs Predicted" />

<h3>🔑 Key Insights</h3>
<ul>
  <li>Vehicle <b>age</b> and <b>kilometers driven</b> have the strongest negative impact on price.</li>
  <li><b>Luxury brands</b> and <b>newer models</b> have higher resale value.</li>
  <li><b>Engine power</b> and <b>mileage</b> significantly affect pricing.</li>
</ul>

<hr>

<h2>🔮 Future Improvements</h2>
<ul>
  <li>Add features like accident history, location, insurance status.</li>
  <li>Use ensemble methods combining XGBoost, Random Forest, and LightGBM.</li>
  <li>Deploy a fully interactive dashboard for visualization & insights.</li>
</ul>

<hr>

<h2>📚 References</h2>
<ul>
  <li><a href="https://www.cardekho.com/">Cardekho Used Car Dataset</a></li>
  <li><a href="https://docs.streamlit.io/">Streamlit Documentation</a></li>
  <li><a href="https://xgboost.readthedocs.io/">XGBoost Documentation</a></li>
</ul>
