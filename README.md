# 💸 Loan Repayment Prediction

## 📘 Project Overview

This project focuses on predicting whether a borrower will fully repay their loan (`loan_paid_back`) using a combination of financial, demographic, and credit-related features. Loan repayment prediction is a **critical task for financial institutions**, as accurate risk assessment helps reduce defaults, optimize lending strategies, and ensure regulatory compliance.  

The dataset contains nearly **600,000 records**, capturing a diverse population of borrowers with a wide range of incomes, credit scores, employment statuses, and loan purposes. By this rich dataset, the project demonstrates **real-world applications of machine learning in finance**, including data preprocessing, feature engineering, model selection, hyperparameter tuning, evaluation, and deployment.  

### 🔍 Key Highlights

- **Comprehensive Data Analysis:** Understanding the relationships between borrower characteristics, loan terms, and repayment behavior.  
- **Feature Engineering:** Creating informative features such as debt-to-income ratios, credit score categories, and interaction terms to improve predictive performance.  
- **Multiple Modeling Approaches:** Comparing traditional machine learning models (Random Forest, XGBoost, LightGBM) with neural networks to identify the best-performing approach.  
- **Explainable AI:** Using feature importance analysis and SHAP values to understand **why the model makes certain predictions**, which is essential for trust and regulatory purposes.  
- **Interactive Deployment:** Providing an end-user interface (via Streamlit) where users can input borrower information and receive a predicted repayment probability in real-time.  

The goal is not only to build an accurate predictive model but also to create a **reproducible and interpretable end-to-end machine learning pipeline**.

By combining **data-driven insights, model performance, and interpretability**, this project highlights the practical application of machine learning in **financial risk assessment** and serves as a strong portfolio piece for data science professionals.  

---

## 🧠 Dataset Information

**Features:**

| Category | Columns |
|-----------|----------|
| Financial | `annual_income`, `debt_to_income_ratio`, `loan_amount`, `interest_rate` |
| Credit | `credit_score`, `grade_subgrade` |
| Demographic | `gender`, `marital_status`, `education_level`, `employment_status` |
| Loan Info | `loan_purpose` |
| Target | `loan_paid_back` |

- **Rows:** ~593,994  
- **Target variable:** `loan_paid_back` (float; 1 = fully repaid, 0 = default or partial)  
- **Data Type:** Mixed numerical and categorical variables  

---

## 🧩 Project Workflow

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical features(One-Hot Encoding, Ordinal Encoding)
   - Feature scaling and normalization
   - Outlier detection (IsolationForest)

2. **Exploratory Data Analysis (EDA)**
   - Visualizing distributions, correlations and mutual information
   - Understanding feature importance
   - Detecting skewness and transformations (Yeo-Johnson)

3. **Feature Engineering**
   - Creating new features such as income-to-loan ratio, risk index, etc.
   - Removing redundant or low-variance features

4. **Model Development**
   - Algorithms tested:
     - Logistic Regression  
     - Random Forest  
     - XGBoost / LightGBM  
     - Neural Network (Keras / TensorFlow)
   - Hyperparameter tuning using **Optuna** and **Keras Tuner**

5. **Model Evaluation**
   - Metrics: AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
   - Cross-validation (K-Fold)
   - Feature importance and interpretability with SHAP

6. **Deployment**
   - Streamlit app for interactive prediction  
   - Visualizing model metrics and feature influences dynamically

---

## 🛠️ Technologies Used

| Category | Libraries |
|-----------|------------|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly`, `streamlit` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep Learning | `TensorFlow`, `Keras`, `PyTorch` |
| Optimization | `optuna`, `keras_tuner` |
| Statistics | `scipy`, `statsmodels` |

---

## 🚀 Future Work

- Integrate external credit history datasets for improved accuracy  
- Add time-series analysis for sequential loan performance prediction  
- Experiment with ensemble stacking models  
- Deploy the app on **Streamlit Cloud** or **AWS EC2**  
- Incorporate explainable AI dashboards using **Plotly Dash** or **Streamlit + SHAP**  

---

## 📈 Example Use Case

A bank analyst inputs the following borrower data into the Streamlit dashboard:

| Feature | Value |
|----------|--------|
| `annual_income` | 72,000 |
| `debt_to_income_ratio` | 0.35 |
| `credit_score` | 680 |
| `loan_amount` | 12,000 |
| `interest_rate` | 7.8 |
| `education_level` | Bachelor |
| `employment_status` | Full-time |


The trained model outputs:  
👉 **Predicted probability of full repayment:** 

---