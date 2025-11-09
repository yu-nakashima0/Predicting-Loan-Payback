import streamlit as st
import sys
import os
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


st.title("Prediction Result")

def add_encoded_features(df):
    feature_list = [
        "id","annual_income","debt_to_income_ratio","credit_score",                        
        "loan_amount","interest_rate",  
        "gender_Female", "gender_Male", "gender_Other" , "marital_status_Divorced", "marital_status_Married" ,      
        "marital_status_Single", "marital_status_Widowed" , "loan_purpose_Business" , "loan_purpose_Car",   
        "loan_purpose_Debt consolidation", "loan_purpose_Education","loan_purpose_Home" ,"loan_purpose_Medical" ,    
        "loan_purpose_Other", "loan_purpose_Vacation","employment_status_Employed",        
        "employment_status_Retired","employment_status_Self-employed","employment_status_Student",
        "employment_status_Unemployed","education_level_encoded","grade_subgrade_encoded" 
    ]

    new_df = pd.DataFrame([{col: 0 for col in feature_list}])
    for i in feature_list:
        for j in df.columns:
            if i == j:
                new_df[i] = df[j]
    return new_df 


#feature engineering
#return : new DataFrame with new features
def feature_engineering(df):
    df["income_to_loan_ratio"] = df["annual_income"] / (df["loan_amount"] + 1)
    df["debt_to_income_ratio_log"] = np.log1p(df["debt_to_income_ratio"])
    df["interest_income_ratio"] = df["interest_rate"] / (df["annual_income"] + 1)
    df["income_x_credit"] = df["annual_income"] * df["credit_score"]
    df["loan_amount_x_interest"] = df["loan_amount"] * df["interest_rate"]
    df["employment_marital"] = (
    df["employment_status_Employed"] * df["marital_status_Married"]
    )
    df["employment_unemployed_and_high_debt"] = (
        df["employment_status_Unemployed"] * (df["debt_to_income_ratio"] > 0.4)
    ).astype(int)
    risky_purposes = ["Debt consolidation", "Medical", "Vacation"]
    df["loan_purpose_risk_group"] = df["loan_purpose_Education"] + df["loan_purpose_Medical"]
    return df

try :
    if "prediction_input" in st.session_state:
        df = st.session_state["prediction_input"]
        st.write("User input data:")
        st.write(df)
        df = add_encoded_features(df)
        print(df.columns)
        df = feature_engineering(df)
        print(df)
        print("feature engineering finished")
        model_path = os.path.join(os.path.dirname(__file__), "../../loan_model.pkl")
        model_path = os.path.abspath(model_path)
        model = joblib.load(model_path)
        #model = joblib.load("../../loan_model.pkl")
        prediction = model.predict(df)
        st.write("Loan paid back? :")
        st.write(prediction)
    else:
        st.warning("no input founded")
except Exception as e:
    st.error(f"Error: {e}")