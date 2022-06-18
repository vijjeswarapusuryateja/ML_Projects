# -*- coding: utf-8 -*-
"""
@author: Vijjeswarapu Surya Teja
"""


##################################### WORK FLOW ################################################


# Step 1:
# Importing essential Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2:
# Importing the given data (train and test)

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Concatenating the train and test data to fill missing values uniformily for entire dataset
data = pd.concat([train_data, test_data], axis=0)

# Step 3:
# Copying data into separate DataFrame to create analysis and features

df = data.copy()


####################### Basic Plot Analysis #####################################################

# Since most of the data is pretty straight forward we will be looking only few plots

# The plot represents an imbalanced dataset. 
# Oversampling can help to predict more potential leads but the idea here is to 
# manage sales bandwidth efficiently so oversampling the data might not be good as the sales team 
# would be calling lot of undesired calls
plt.figure()
sns.countplot(df['buy'])

# Buys almost equal distributed among 3,4,5 campaigns
plt.figure()
sns.countplot(df['campaign_var_1'], hue=df['buy'])

# No outliers detected in campaign_var_1
plt.figure()
sns.boxplot(df['campaign_var_1'])

# Buys almost equal distributed among 4,5,6 campaigns
plt.figure()
sns.countplot(df['campaign_var_2'], hue=df['buy'])

# No outliers detected in campaign_var_2
plt.figure()
sns.boxplot(df['campaign_var_2'])

# 'products_purchased' doesn't give much info as there were more missing values in the data
plt.figure()
sns.countplot(df['products_purchased'], hue=df['buy'])


################## PART A (Data Cleaning and Feature Engineering)##################################



# Step 1:
# Identfying the datatypes and columns in data

data_types = df.dtypes
cols = df.columns


# Step 2
# Dropping obviously unwanted columns
df = df.drop('id', axis=1)


# Step 3:
# Looking for missing values in the data

missing_vals = df.isnull().sum(axis=0)
# 'products_purchased' column has 55.4% missing values
# 'signup_date' column has 41.5% missing values

# Dropping 'products_purchased' column as imputing may lead to erroneous model
df = df.drop('products_purchased', axis=1)

# Step 4:
# Imputing missing values

# Changing date columns for further simplification
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['created_at'] = pd.to_datetime(df['created_at'])

# Creating a new column from 'signup_date' column
df['user_signup'] = df['signup_date']

# Imputing missing values in 'signu_date; column with lead 'created_at' column
df['signup_date'] = df['signup_date'].fillna(df['created_at'])

# Dropping 'created_at' column as it may lead to erroneous model
df = df.drop('created_at', axis=1)


# Step 5:
# Creating New columns from exisiting columns

# new columns from 'signup_column'
df['signup_day'] = df['signup_date'].dt.day
df['signup_month'] = df['signup_date'].dt.month
df['signup_year'] = df['signup_date'].dt.year

# Dropping 'signup_date' column because it has been split into multiple columns
df = df.drop('signup_date', axis=1)

# Creating a new activity column from previously cloned column
df['signup_activity'] = df['user_signup'].dt.year
df['user_activity_var_0'] = df['signup_activity']

# Filling missing values with 'zero' indicating inactiveness
df['user_activity_var_0'] = df['user_activity_var_0'].fillna(0)

# Changing exisitng signup information into activity and making as 'One'
df.loc[df['user_activity_var_0'] > 2000, 'user_activity_var_0'] = 1

# Dropping 'user_signup' and 'signup_activity' because 'user_activity_var_0 is similar
df = df.drop(['user_signup', 'signup_activity'], axis=1)


# Step 6:
# Creating New columns from combination of columns

# Summing up all campaigns into single column
df['total_campaigns'] = df['campaign_var_1'] + df['campaign_var_2']

df.columns
# Summing up all user activity variables into single activity
df['total_activity'] = df.iloc[:, 2:14].sum(axis=1)
df['total_activity'] = df['total_activity'] + df['user_activity_var_0']


# Step 7:
# Checking for multicollinearity among the features

# Generating correlation matrix for dataset
corr_matrix = df.corr()

# 'campaign_var_1' has 0.91 correlation with 'total_campaigns'
# 'campaign_var_2' has 0.85 correlation with 'total_campaigns'
# 'signup_day' has 0.91 correlation with 'total_activity'

# Dropping below columns due to strong correlation effect
df = df.drop(['campaign_var_1', 'campaign_var_2', 'signup_day', ], axis=1)








######################## PART B (Data Pre-Processing)#####################################


# Step 1:
# Copying data into separate DataFrame for Data Pre-Processing
features_set = df.copy()

# Step 2:
# Separating trainset dataset for further data pre-processing

train_set = features_set.iloc[:len(train_data), :]
test_set = features_set.iloc[len(train_data):, :]


# Step 3:
# Creating 'X_t' and 'Y_t' dataset from trainset for further analysis

X_t = train_set.drop('buy', axis=1)
Y_t = train_set['buy']
Xt_cols = X_t.columns

# Step 4:
# Selecting best features with statistical methods

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_t = mms.fit_transform(X_t)

# Statistical Analysis using Chisqaure
from sklearn.feature_selection import SelectKBest, chi2
kbest = SelectKBest(chi2, k='all')
kbest.fit_transform(X_t, Y_t)
pvalues = kbest.pvalues_

# Null hypothesis: 'Y' does not depend on 'X' features
    # At 95% confidence interval if p < 0.05 we reject the null hypothesis for 'X' features
    # For 'user_activity_var_3' the p_value is 0.33 so it can be dropped
    # For 'user_activity_var_6' the p_value is 0.32 so it can be dropped

# Selecting features with p < 0.05
pcols = pd.DataFrame(pvalues).rename(columns={0: 'p_cols'})
select_features = pcols[pcols['p_cols'] < 0.05]
final_features_cols = select_features.index.values.tolist()


# Step 5:
# Preparing the final datasets for the model

X = train_set.drop('buy', axis=1).iloc[:, final_features_cols]
Y = train_set['buy']



# Step 6:
# Splitting the datasets for model evaluation

from sklearn.model_selection import train_test_split

# Using 70% of data for training and 30 % for testing and stratifying on Y
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)


# Step 7:
# Scaling the data before applying the model

mms2 = MinMaxScaler()
x_train = mms2.fit_transform(x_train)
x_test = mms2.transform(x_test)
X = mms2.fit_transform(X)



#################################### PART C (Model Selection)###################################


# Since this is classification problem the following classification models give best results
    # RandomForest
    # XGBoost
    # LightGBM
    # CatBoost (works best if more categorical features are present: can be dropped for this problem)

# Importing all models as mentioned above

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Step 1:
# Using GridSearchCV to find the model that gives better results

# Calling the models
rfc = RandomForestClassifier()
xgb = XGBClassifier()
lgb = LGBMClassifier()


# Creating Params for hyperparameter tuning

rfc_params = {
                'max_depth': [10, 40, 80],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [200, 400, 800]
             }



xgb_params = {
                'max_depth':[4,6,8],
                'learning_rate':[0.01, 0.04, 0.1],
                'n_estimators':[100, 1000, 2000],
                'colsample_bytree':[0.6, 0.8, 1],
                'subsample':[0.8, 1],
                'gamma':[0,1]
             }



lgb_params = {
                'max_depth':[4,6,8],
                'num_leaves':[16, 64, 256],
                'learning_rate':[0.01, 0.04, 0.1],
                'num_iterations':[100, 1000, 2000],
                'feature_fraction':[0.6, 0.8, 1]
             }




# Step 2:
# Performing GridSearchCV for selecting best model

# Creating scorer for GridSearchCV
from sklearn.metrics import f1_score, make_scorer
f1_scorer = make_scorer(f1_score)

# Importing GridSearchCV from sklearn library
from sklearn.model_selection import GridSearchCV

# RandomForest GridSearch
gsc_rfc = GridSearchCV(estimator=rfc, param_grid=rfc_params, cv=4, scoring=f1_scorer)
gsc_fit_rfc = gsc_rfc.fit(X, Y)
cv_results_rfc = pd.DataFrame.from_dict(gsc_fit_rfc.cv_results_)

# XGBoost GridSearch
gsc_xgb = GridSearchCV(estimator=xgb, param_grid=xgb_params, cv=4, scoring=f1_scorer)
gsc_fit_xgb = gsc_xgb.fit(X, Y)
cv_results_xgb = pd.DataFrame.from_dict(gsc_fit_xgb.cv_results_)

# LGBM GridSearch
gsc_lgb = GridSearchCV(estimator=lgb, param_grid=lgb_params, cv=4, scoring=f1_scorer)
gsc_fit_lgb = gsc_lgb.fit(X, Y)
cv_results_lgb = pd.DataFrame.from_dict(gsc_fit_lgb.cv_results_)


# GridSearchCV Results: 
    # XGBoost with below params gave better results compared to RandomForest and LightGBM
    # max_depth=4, learning_rate=0.01, n_estimators=2000, colsample_bytree=0.7, subsample=1, gamma=0


# Step 3:
# Using best model for training and testing

# Calling and fitting the model with selected parameters
xgb_test = XGBClassifier(max_depth=4, objective='binary:logistic', learning_rate=0.01, n_estimators=2000, colsample_bytree=0.7)
xgb_test.fit(x_train, y_train)

# Predicting the results
y_predict_xgb = xgb_test.predict(x_test)
y_proba_xgb = xgb_test.predict_proba(x_test)
fi_xgb = xgb_test.feature_importances_



# Step 4:
# Evaluating the predicted results with metrics

# Importing metrics from sklearn Library
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
cm_xgb = confusion_matrix(y_test, y_predict_xgb)
f1_xgb = f1_score(y_test, y_predict_xgb)
accu_xgb = accuracy_score(y_test, y_predict_xgb)
cr = classification_report(y_test, y_predict_xgb)

# Results: accuracy_score=0.97, f1_score=0.71




################################# PART D (Running Model on Test Data)####################################


# Step 1:
# Running the model to predict results for test dataset

# Calling the model with parameters
xgb_main = XGBClassifier(max_depth=6, objective='binary:logistic', learning_rate=0.01, n_estimators=4000, colsample_bytree=0.8)
xgb_main.fit(X, Y)

# Preparing the test set with selected features
test_set = test_set.drop('buy', axis=1).iloc[:, final_features_cols]

# Scaling the test set
test_set = mms2.transform(test_set)

# Predicitng the results
test_predict_xgb = xgb_test.predict(test_set)


# Step 2:
# Exporting the Predicted Results

# Gathering ids from sample submision data
sample_data = pd.read_csv('sample_submission.csv')
sample_data = sample_data.drop('buy', axis=1)

# adding ids to the predicted results
test_results_xgb = pd.DataFrame(test_predict_xgb).rename(columns={0: 'buy'})
final_results_xgb= pd.concat([sample_data, test_results_xgb], axis=1)

# Exporting the final results for submission
export = final_results_xgb.to_csv('av_jthon_result_test.csv', index=False)











































