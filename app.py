import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from sklearn.feature_selection import mutual_info_classif



"""
encoding categorical variables
 6   gender                593994 non-null  object -> One-Hot Encoding
 7   marital_status        593994 non-null  object -> One-Hot Encoding
 8   education_level       593994 non-null  object -> Ordinal Encoding
 9   employment_status     593994 non-null  object -> One-Hot Encoding
 10  loan_purpose          593994 non-null  object -> One-Hot Encoding
 11  grade_subgrade        593994 non-null  object -> Ordinal Encoding
return: dataframe with encoded categorical variables
"""
def encode_categorical_variables(df):
    df = pd.get_dummies(
        df, 
        columns = ["gender", "marital_status", "loan_purpose", "employment_status"], 
        drop_first = False,
        dtype="int64"
    )
    """
    for i in df["education_level"].unique():
        print(i)
    """
    encoder = OrdinalEncoder(categories=[["High School", "Bachelor's", "Master's", "PhD", "Other"]])
    df["education_level_encoded"] = encoder.fit_transform(df[["education_level"]])
    """
    for i in df["grade_subgrade"].unique():
        print(i)
    """
    encoder = OrdinalEncoder(
        categories=[[
            'A1', 'A2', 'A3', 'A4', 'A5',
            'B1', 'B2', 'B3', 'B4', 'B5',
            'C1', 'C2', 'C3', 'C4', 'C5',
            'D1', 'D2', 'D3', 'D4', 'D5',
            'E1', 'E2', 'E3', 'E4', 'E5',
            'F1', 'F2', 'F3', 'F4', 'F5'
    ]])
    df["grade_subgrade_encoded"] = encoder.fit_transform(df[["grade_subgrade"]])
    df.drop(columns=["education_level", "grade_subgrade"], inplace=True)
    print(df.info())
    print(df.head())
    return df

"""
handling missing values
return: dataframe with missing values handled
"""
def handle_missing_values(df):
    missing_values = df.isnull().sum()
    if missing_values.sum() != 0:
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                df[column].interpolate(method='quadratic')
    return df


"""
normalizing numerical features
return: dataframe with normalized numerical features
"""
def normalize_numerical_features(df):
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df


"""
outlier detection and removal
return: dataframe with outliers removed
"""
def remove_outliers(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


"""
make grouped feature list
return: dictionary of feature groups
"""
def make_grouped_feature_list(df):
    feature_groups = {
        "Financial":["annual_income","debt_to_income_ratio","loan_amount","interest_rate"],
        "Credit": ["credit_score","grade_subgrade"],
        "Demographic": ["gender", "marital_status","education_level","employment_status"],
        "Loan": ["loan_purpose"],
        "Target":["loan_paid_back"]
    }

    return feature_groups


"""
visualization each distribution
"""
def visualize_data_boxplot(df, feature_groups, unique_key):
    st.title("Boxplots by feature group")
    group = st.selectbox("select a group", feature_groups.keys(), key=unique_key)
    cols = feature_groups[group]
    st.subheader(f"Boxplots for groups: {group}")
    for column in cols:
        fig, ax = plt.subplots()
        ax.boxplot(df[column].dropna())
        ax.set_title(column)
        st.pyplot(fig)


"""
visualization each number of counts
"""
def visualize_data_countplot(df, feature_groups, unique_key):
    st.title("Countplots by feature group")
    group = st.selectbox("select a group", feature_groups.keys(), key=unique_key)
    cols = feature_groups[group]
    st.subheader(f"Countplots for groups: {group}")
    for column in cols:
        fig, ax = plt.subplots()
        sns.countplot(x=df[column], ax=ax)
        ax.set_title(column)
        st.pyplot(fig)


"""
visualization correlation heatmap, mutual information
"""
def visualize_more_info(df):
    df.drop(columns = "id",inplace=True)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)

    X = df.drop(columns=["loan_paid_back"])
    y = df["loan_paid_back"]
    mi = mutual_info_classif(X,y, discrete_features='auto', random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=mi_scores.values, y=mi_scores.index, ax=ax2)
    plt.title("Mutual Information Scores")
    st.pyplot(fig2)




df = pd.read_csv('train.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)

df_encoded = encode_categorical_variables(df)
df_encoded = handle_missing_values(df_encoded)
df_encoded = normalize_numerical_features(df_encoded)
df_encoded = remove_outliers(df_encoded)

feature_groups = make_grouped_feature_list(df)

#visualize_data_boxplot(df, feature_groups, "before_encoding")
#visualize_data_countplot(df, feature_groups, "before_encoding_countplot")
visualize_more_info(df_encoded)