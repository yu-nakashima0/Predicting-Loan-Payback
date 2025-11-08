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
from xgboost import plot_importance
import optuna
from xgboost.callback import EarlyStopping
from sklearn.feature_selection import SelectFromModel


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


"""
check correlation
"""
def visualize_correlation(df):
    #df = df.drop(columns = ["id"])
    corr = df.corr()
    print(corr)
    sns.heatmap(corr, cmap = "coolwarm", center = 0)
    plt.show()

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
    list_for_importance = []
    
    for train_index, test_index in sss.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        #model = logistic_regression_model(X_train, y_train)
        #model = random_forest_model(X_train, y_train)
        model, importance = xgboost_model(X_train, y_train)
        list_for_importance.append(importance)
        #model = neural_network_model(X_train, y_train)
        #y_pred_proba = model.predict(X_test).ravel()
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        scores.append(auc)
    
    print(f"{k}-Fold Cross-Validation AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores, list_for_importance

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
    model.fit(
        X_train, y_train,     
        verbose=True                     
    )
    selection = SelectFromModel(model, threshold="median", prefit=True)
    X_selected = selection.transform(X)
    importance = model.get_booster().get_score(importance_type='gain')
    importance = pd.DataFrame(
        importance.items(), columns=['Feature', 'Importance']
    ).sort_values(by='Importance', ascending=False)
    print(importance)
    return model, importance,X_selected


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
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    return model



"""
combine feature importance from all folds and calculate mean importance
return: mean feature importance dataframe
"""
def statistik_feature_importance(list_for_importance):
    combined_importance = pd.concat(list_for_importance)
    mean_importance = combined_importance.groupby('Feature')['Importance'].mean().reset_index()
    mean_importance = mean_importance.sort_values(by='Importance', ascending=False)
    print(mean_importance)


"""
hyperparameter tuning with Optuna
return : average AUC
"""
def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'random_state': 42,
        'early_stopping_rounds': 10,
        'scale_pos_weight' : 4
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBClassifier(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)]
        )
        y_pred = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_pred)
        print(f"AUC for fold: {auc}")
        aucs.append(auc)

    return np.mean(aucs)


"""
hyperparameter tuning with Optuna
return: best model, best hyperparameters, best AUC, feature importance dataframe
"""
def tune_xgboost_with_optuna(df, n_trials):
    X = df.drop(columns=['loan_paid_back'])
    y = df['loan_paid_back']

    # Optuna Study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("Best AUC:", study.best_value)
    print("Best hyperparameters:", study.best_params)

    # Trainiere Modell mit den besten Parametern auf gesamten Datensatz
    best_params = study.best_params
    best_params.update({'use_label_encoder': False, 'eval_metric': 'auc', 'random_state': 42})

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X, y)

    # Feature Importance
    importance = best_model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

    print("Top Features:\n", importance_df)
    print("Best Hyperparameters:\n", study.best_params)
    print("Best AUC:\n", study.best_value,)

    return best_model, study.best_params, study.best_value, importance_df



df = pd.read_csv('train.csv')
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)
print(df['loan_paid_back'].value_counts())

df_encoded = encode_categorical_variables(df)
df_encoded = handle_missing_values(df_encoded)


#visualize_data_boxplot(df, feature_groups, "before_encoding")
#visualize_data_countplot(df, feature_groups, "before_encoding_countplot")
#visualize_more_info(df_encoded)

#df_encoded = transform_skewed_features(df_encoded)
#df_encoded = feature_engineering(df_encoded) -> implement after baseline


#scores, list_for_importance =  k_fold_cross_validation(df_encoded, 5)

#statistik_feature_importance(list_for_importance)

df_encoded = feature_engineering(df_encoded)
print(df_encoded.columns)
df_encoded = normalize_numerical_features(df_encoded)
#visualize_correlation(df_encoded)
#df_encoded = remove_outliers(df_encoded)
print(df_encoded['loan_paid_back'].value_counts())
feature_groups = make_grouped_feature_list(df)
best_model, best_params, best_auc, importance_df = tune_xgboost_with_optuna(df_encoded, n_trials=10)

