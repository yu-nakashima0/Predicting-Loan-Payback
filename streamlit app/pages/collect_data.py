import streamlit as st
import pandas as pd


st.title("Please enter the data required for forecasting.")

# data that i need to correct:             
#     (feature)              (datatype)   (example)
# 1   annual_income          float64      29367.99
# 2   debt_to_income_ratio   float64      0.084
# 3   credit_score           int64        736
# 4   loan_amount            float64      2528.42
# 5   interest_rate          float64      13.67
# 6   gender                 object       Female
# 7   marital_status         object       Single
# 8   education_level        object       High School
# 9   employment_status      object       Self-employed
# 10  loan_purpose           object       Other
# 11  grade_subgrade         object       C3
# 12  loan_paid_back         float64      1.0


annual_income = st.number_input("Enter your annual income ($):", min_value = 0.0, format= "%.2f")
debt_to_income_ratio = st.number_input("Enter your dept to income ratio:", min_value = 0.0, max_value = 1.0, format= "%.2f")
credit_score = st.number_input("Enter your credit score:", min_value = 0)
loan_amount = st.number_input("Enter your loan amount:", min_value = 0.0)
interest_rate = st.number_input("Enter your interest rate:", min_value = 0.0)
gender = st.radio("Select your gender:",["Female","Male","Other"])
marital_status = st.radio("Select your marital status:", ['Single' ,'Married', 'Divorced', 'Widowed'])
education_level = st.radio("Select your education level:", ['High School', "Bachelor's", "Master's", 'PhD', 
'Other'])
employment_status = st.radio("Select your employment status:", ['Self-employed', 'Employed', 'Unemployed', 'Retired', 'Student'])
loan_purpose = st.radio("Select your loan purpose:", ['Debt consolidation', 'Home', 'Education', 'Vacation', 'Car', 
 'Medical', 'Business','Other'])

list_of_grade = []
for i in ["A","B","C","D","E","F"]:
    for j in ["1","2","3","4","5"]:
        grade = i + j
        list_of_grade.append(grade)

grade_subgrade = st.radio("Select your grade and subgrade:", list_of_grade)


if st.button("send"):
    try:
        data = {
            'annual_income':[annual_income], 
            'debt_to_income_ratio':[debt_to_income_ratio], 
            'credit_score':[credit_score],
            'loan_amount':[loan_amount], 
            'interest_rate':[interest_rate],
            'gender':[gender], 
            'marital_status':[marital_status],
            'education_level':[education_level], 
            'employment_status':[employment_status], 
            'loan_purpose':[loan_purpose],
            'grade_subgrade':[grade_subgrade]
        }
        df = pd.DataFrame(data)
        st.session_state["prediction_input"] = df
        st.switch_page("pages/predicting.py")
    except Exception as e:
        st.error(f"Error: {e}")






