# 🌍 Global Pollution Analysis: AI-Powered Clustering and Prediction for Energy Recovery

This project explores how pollution data from multiple countries correlates with energy recovery potential. Using unsupervised learning (K-Means and Hierarchical Clustering) and regression modeling (Neural Networks & Linear Regression), we segment countries based on environmental indicators and predict how pollution impacts sustainable energy practices.

## 🚀 Project Overview

- **Domain:** Environment & Energy
- **Tech Stack:** Python, Pandas, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn
- **Problem Statement:**  
  Analyze global pollution metrics and predict energy recovery potential using machine learning models.

## 🎯 Objectives

- Cluster countries based on pollution indicators (Air, Water, Soil).
- Predict energy recovery using neural networks and linear regression.
- Derive actionable environmental insights from unsupervised learning.
- Compare clustering and prediction models for performance and interpretability.

## 🗂️ Dataset

- **Name:** `Global_Pollution_Analysis.csv`
- **Fields include:** Country, Year, Air/Water/Soil Pollution Indices, CO₂ Emissions, Industrial Waste, Population, Energy Recovered (in GWh), Energy Consumption per Capita.

## 🧪 Machine Learning Phases

### Phase 1 - Data Preprocessing & Feature Engineering
- Handled missing values
- Label Encoding for countries
- Feature scaling (StandardScaler)
- Created new metrics like average pollution index

### Phase 2 - Clustering Models
- 📌 **K-Means:** Clustered countries based on pollution & energy variables.  
- 🌳 **Hierarchical Clustering:** Revealed nested similarities using dendrograms.

### Phase 3 - Neural Network Prediction
- Feedforward neural network to predict `Energy_Recovered (in GWh)`
- Compared with Linear Regression
- Evaluation metrics: R², MSE, MAE

## 📊 Results

| Model                | R² Score | MSE     | MAE     |
|---------------------|----------|---------|---------|
| Neural Network       | -0.23    | 1.39    | 1.01    |
| Improved Neural Net  | -0.45    | 1.63    | 1.06    |
| Linear Regression    | **-0.04**| **1.17**| **0.97**|

📌 **Insight:** Linear regression outperformed neural networks on this dataset. Clustering revealed strong grouping trends based on pollution severity and energy recovery potential.

## 💡 Key Takeaways

- Countries with high pollution showed lower energy recovery, needing urgent intervention.
- Clustering helped benchmark countries and suggest peer-level improvement strategies.
- Simpler models often perform better when data is small and linear.

## 🧠 Future Scope

- Use larger datasets with richer geographical and policy-level features.
- Implement ensemble models like XGBoost or Random Forest for better predictions.
- Deploy as a dashboard using Streamlit or Flask for policy-makers.

## 🔧 Setup Instructions

1. Clone this repository  
```bash
git clone https://github.com/Tushar22-tutu/global-pollution-energy-analysis.git
cd global-pollution-energy-analysis
