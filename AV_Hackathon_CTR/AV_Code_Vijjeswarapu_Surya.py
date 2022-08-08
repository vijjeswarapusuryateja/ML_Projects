# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 18:52:56 2022

@author: -Surya
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


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Concatenating the train and test data to fill missing values uniformily for entire dataset

data = pd.concat([train_data, test_data], axis=0)

# Step 3:
# Copying data into separate DataFrame to create analysis and features

df = data.copy()

####################### Basic Plot Analysis #####################################################

# Since most of the data is pretty straight forward we will be looking only few plots
plt.figure()
sns.histplot(df['click_rate'])


plt.figure()
df.hist()
plt.tight_layout()

################## PART A (Data Cleaning and Feature Engineering)##################################



# Step 1:
# Identfying the datatypes and columns in data


cols = df.columns

info = df.describe()

data_types = df.dtypes

# Step 2:
# Looking for missing values in the data

missing_vals = df.isnull().sum(axis=0)

# Step 3:
# Dropping obviously unwanted columns

df = df.drop(['is_timer', 'campaign_id'], axis=1)


# Step 4:
# Creating new columns from existing columns

df['para_count'] = df['body_len'] / df['mean_paragraph_len']
df['CTA_ratio'] = df['no_of_CTA'] / df['body_len']
df['total_CTA_char'] = df['no_of_CTA'] * df['mean_CTA_len']
df['total_mail_char'] = df['subject_len'] + df['body_len'] + df['total_CTA_char']

# Step 5:
# Changing columns with boolean type to category as oer given problem statement 

df['is_weekend'] = df['is_weekend'].astype('category')
df['is_personalised'] = df['is_personalised'].astype('category')
df['is_urgency'] = df['is_urgency'].astype('category')
df['is_discount'] = df['is_discount'].astype('category')
df['times_of_day'] = df['times_of_day'].astype('category')
df['day_of_week'] = df['day_of_week'].astype('category')
df['target_audience'] = df['target_audience'].astype('category')


# Step 5:
# Changing other similar columns with boolean type to category

df.loc[df['is_price'] > 0, 'is_price'] = 1
df['is_price'] = df['is_price'].astype('category')

df.loc[df['is_image'] > 0, 'is_price'] = 1
df['is_image'] = df['is_image'].astype('category')

df.loc[df['is_quote'] > 0, 'is_price'] = 1
df['is_quote'] = df['is_quote'].astype('category')

df.loc[df['is_emoticons'] > 0, 'is_price'] = 1
df['is_emoticons'] = df['is_emoticons'].astype('category')

df['sender'].value_counts()
df.loc[df['sender'] == 3, 'sender'] = 1
df.loc[df['sender'] != 3, 'sender'] = 0
df['sender'] = df['sender'].astype('category')


# Step 6:
# Selecting numerical and category columns

col_dict = df.dtypes.apply(lambda x: x.name).to_dict()

cat_cols, num_cols = [], []

for key in col_dict.keys():
    if col_dict[key] == 'category':
        cat_cols.append(key)
    else:
        num_cols.append(key)
        
        
        
# Step 7:
# Checking for multicollinearity among the features
# Generating correlation matrix for dataset
# Dropping columns with more than 0.95 collinearity

corr_matrix = df.corr()

df = df.drop('body_len', axis=1)



######################## PART B (Data Pre-Processing)#####################################


# Step 1:
# Copying data into separate DataFrame for Data Pre-Processing

features_set = df.copy()

# Below line was commented as I have decided to use catboost
#features_set = pd.get_dummies(features_set, drop_first=True)

# Step 2:
# Separating trainset dataset for further data pre-processing

train_set = features_set.iloc[:len(train_data), :]

test_set = features_set.iloc[len(train_data):, :]


# Step 3:
# Creating 'X' and 'Y' dataset from trainset for further analysis

X = train_set.drop('click_rate', axis=1)
Y = train_set['click_rate']

X_cols = X.columns




# Step 4:
# Splitting the datasets for model evaluation

# Using 70% of data for training and 30 % for testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)



# Below line was commented as I have decided to use catboost
#from sklearn.preprocessing import MinMaxScaler

#sc = MinMaxScaler()

#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)
#X = sc.fit_transform(X)


#################################### PART C (Model Selection)###################################


# Since this is Regression problem the following Regression models give best results
    # XGBoost
    # LightGBM
    # CatBoost (works best if more categorical features are present)

# Importing all models as mentioned above

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Step 1:
# Using GridSearchCV to find the model that gives better results

# Calling the models
xgb = XGBRegressor()
lgb = LGBMRegressor()
cb = CatBoostRegressor()



# Importing GridSearchCV from sklearn library and making the r2_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, make_scorer
r2_scorer = make_scorer(r2_score)


# Step 2:
# Creating the params

xgb_params = {
                'learning_rate':[0.01, 0.04, 0.06, 0.1],
                'n_estimators':[100, 500, 1000],
                'max_depth':[4,6]
             }


lgb_params = {
                'max_depth':[4,5,6],
                'learning_rate':[0.01, 0.04, 0.06, 0.1],
                'n_estimators':[100, 500, 1000],
             }


cb_params = {
                'depth': [4, 6],
                'iterations': [500, 1000, 2000],
                'learning_rate' : [0.01, 0.1, 0.04, 0.06]
            }



# Step 3:
# Performing Gridsearch

# XGBoost GridSearch (uncomment dummies in PART B-Step 1 )
gsc_xgb = GridSearchCV(estimator=xgb, param_grid=xgb_params, cv=4, scoring=r2_scorer, return_train_score=True)
gsc_fit_xgb = gsc_xgb.fit(X, Y)
cv_results_xgb = pd.DataFrame.from_dict(gsc_fit_xgb.cv_results_)

# LightBGM GridSearch (uncomment dummies in PART B-Step 1 )
gsc_lgb = GridSearchCV(estimator=lgb, param_grid=lgb_params, cv=4, scoring=r2_scorer, return_train_score=True)
gsc_fit_lgb = gsc_lgb.fit(X, Y)
cv_results_lgb = pd.DataFrame.from_dict(gsc_fit_lgb.cv_results_)

# CatBoost GridSearch (uncomment dummies in PART B-Step 1 )
gsc_cb = GridSearchCV(estimator=cb, param_grid=cb_params, cv=4, scoring=r2_scorer, return_train_score=True)
gsc_fit_cb = gsc_cb.fit(X, Y)
cv_results_cb = pd.DataFrame.from_dict(gsc_fit_cb.cv_results_)

# GridSearchCV Results: 
    # Catboost with below params gave better results compared to XGboost and LightGBM
    # Option 1): cat_features=cat_cols, depth=4, iterations=1000, learning_rate=0.01
    # Option 2): cat_features=cat_cols, depth=4, iterations=1000, learning_rate=0.06
    # I have decided to go with option 1 as it was giving slighlty better score compared to option 2



# Step 4:
# Using best model for training and testing
# Calling and fitting the model with selected parameters

from catboost import CatBoostRegressor

cb_train = CatBoostRegressor(cat_features=cat_cols, depth=4, iterations=1000, learning_rate=0.1)

cb_train.fit(x_train, y_train)


# Predicting the results
y_predict_cb = cb_train.predict(x_test)

score_cb = cb_train.score(x_test, y_test)

fi_cb = cb_train.feature_importances_

r2_cb = r2_score(y_test, y_predict_cb)

# Results: r2_score=0.54




from xgboost import XGBRegressor

xgb = XGBRegressor(max_depth=4, learning_rate=0.01, n_estimators=500, colsample_bytree=0.6, subsample=0.8)

#xgb.fit(x_train, y_train)

xgb.fit(X, Y)

y_predict_xgb = xgb.predict(x_test)

score_xgb = xgb.score(x_test, y_test)

r2_xgb = r2_score(y_test, y_predict_xgb)




from lightgbm import LGBMRegressor

lgb = LGBMRegressor(max_depth=4, learning_rate=0.06, n_estimators=500)

#lgb.fit(x_train, y_train)

lgb.fit(X, Y)

y_predict_lgb = lgb.predict(x_test)

score_lgb = lgb.score(x_test, y_test)

r2_lgb = r2_score(y_test, y_predict_lgb)







################################# PART D (Running Model on Test Data)####################################


# Step 1:
# Running the model to predict results for test dataset

# Calling the model with parameters
cb_test = CatBoostRegressor(cat_features=cat_cols, depth=4, iterations=1000, learning_rate=0.1)

cb_test.fit(X, Y)

# Preparing the test set
test_set = test_set.drop('click_rate', axis=1)


# Predicitng the results
test_predict = cb_test.predict(test_set)
test_results = pd.DataFrame(test_predict).rename(columns={0: 'click_rate'})

# Step 2:
# Exporting the Predicted Results

# Gathering ids from sample submision data
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission = sample_submission.drop('click_rate', axis=1)

# adding ids to the predicted results
final_results = pd.concat([sample_submission, test_results], axis=1)

# Exporting the final results for submission
final_csv = final_results.to_csv('av_jthon_result.csv', index=False)





















