import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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
    target = df["loan_paid_back"]
    df = df.drop(columns=["loan_paid_back"])
    scaler = MinMaxScaler()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    df["loan_paid_back"] = target
    return df


"""
outlier detection and removal
return: dataframe with outliers removed
"""
def remove_outliers(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for column in numerical_columns:
        if column != "loan_paid_back":
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
    corr_scores = df.corr()['loan_paid_back'].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x=corr_scores.values, y=corr_scores.index, ax=ax)
    plt.title("Correlation Scores")
    plt.xlim(-1, 1)  
    st.pyplot(fig)

    X = df.drop(columns=["loan_paid_back"])
    y = df["loan_paid_back"]
    mi = mutual_info_classif(X,y, discrete_features='auto', random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=mi_scores.values, y=mi_scores.index, ax=ax2)
    plt.title("Mutual Information Scores")
    st.pyplot(fig2)


"""
detect skewness and transform only the skewed features
return : dataframe with transformed distributions
"""
def transform_skewed_features(df):
    skewness = df[df.columns].apply(lambda x: skew(x.dropna()))
    skewed_features = skewness[abs(skewness) > 0.5].index
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    df[skewed_features] = pt.fit_transform(df[skewed_features])
    return df


"""
create new features and felete redundant features
return: dataframe with engineered features
"""
def feature_engineering(df):
    ## implement after baseline model
    return df



"""
k-fold cross validation
return: average AUC
"""
def k_fold_cross_validation(df, k):
    X = df.drop(columns=["loan_paid_back"])
    y = df["loan_paid_back"]
    
    #skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scores = []
    
    for train_index, test_index in sss.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        #model = logistic_regression_model(X_train, y_train)
        #model = random_forest_model(X_train, y_train)
        #model = xgboost_model(X_train, y_train)
        model = neural_network_model(X_train, y_train)
        y_pred_proba = model.predict(X_test).ravel()
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        """
        auc = roc_auc_score(y_test, y_pred_proba)
        scores.append(auc)
    
    print(f"{k}-Fold Cross-Validation AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores

"""
logistic regression model
return: model
"""
def logistic_regression_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=20
    )
    model.fit(X_train, y_train)
    return model


"""
random forest model
return: model
"""
def random_forest_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=20, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


"""
XGBoost models
return: model
"""
def xgboost_model(X_train, y_train):
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


"""
Neural Network model
return: model
"""
def neural_network_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    return model

df = pd.read_csv('train.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)
print(df['loan_paid_back'].value_counts())

df_encoded = encode_categorical_variables(df)
df_encoded = handle_missing_values(df_encoded)
df_encoded = normalize_numerical_features(df_encoded)
#df_encoded = remove_outliers(df_encoded)
print(df_encoded['loan_paid_back'].value_counts())
feature_groups = make_grouped_feature_list(df)

#visualize_data_boxplot(df, feature_groups, "before_encoding")
#visualize_data_countplot(df, feature_groups, "before_encoding_countplot")
visualize_more_info(df_encoded)

#df_encoded = transform_skewed_features(df_encoded)
#df_encoded = feature_engineering(df_encoded) -> implement after baseline


k_fold_cross_validation(df_encoded, 5)



