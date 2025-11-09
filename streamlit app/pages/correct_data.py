import streamlit as st

st.write("hello")

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
education_level = st.radio("Select your education level:", ['High School', "Master's", "Bachelor's", 'PhD', 
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
        st.switch_page("pages/predicting.py")
    except Exception as e:
        st.error(f"Error: {e}")