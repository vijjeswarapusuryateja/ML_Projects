# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 00:39:35 2022

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

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Concatenating the train and test data to fill missing values uniformily for entire dataset
data = pd.concat([train_data, test_data], axis=0)

# Step 3:
# Copying data into separate DataFrame to create analysis and features

df = data.copy()



################## PART A (Data Cleaning and Feature Engineering)##################################



# Step 1:
# Identfying the datatypes and columns in data

data_types = df.dtypes
cols = df.columns


# Step 2
# Dropping obviously unwanted columns
df = df.drop('row_id', axis=1)


# Step 3:
# Looking for missing values in the data

percent_missing = df.isnull().sum(axis=0) * 100 / len(df)


count_of_columns_removed = 0

for i in df.columns:

    percent_NA = round(100*(df[i].isnull().sum()/len(df.index)),2)     


    if percent_NA >= 50:
        print(i)
        df.drop(columns=i, inplace=True)
        count_of_columns_removed += 1

print(count_of_columns_removed)


new_cols = df.columns

change_cols = ['scout_id', 'winner', 'team', 'competitionId', 'player_position_1', 'player_position_2']

df[change_cols] = df[change_cols].astype('object')





col_dict = df.dtypes.apply(lambda x: x.name).to_dict()

cat_cols, num_cols = [], []

for key in col_dict.keys():
    if col_dict[key] == 'object':
        cat_cols.append(key)
    else:
        num_cols.append(key)
        
df[num_cols] = df[num_cols].fillna(df.mean().iloc[0])

df[cat_cols] = df[cat_cols].fillna(df.mode().iloc[0])







######################## PART B (Data Pre-Processing)#####################################


# Step 1:
# Copying data into separate DataFrame for Data Pre-Processing
df['rating_num'] = data['rating_num']
features_set = df.copy()
features_set = pd.get_dummies(features_set, drop_first=True)

# Step 2:
# Separating trainset dataset for further data pre-processing

train_set = features_set.iloc[:len(train_data), :]
test_set = features_set.iloc[len(train_data):, :]


# Step 3:
# Creating 'X_t' and 'Y_t' dataset from trainset for further analysis

X_t = train_set.drop('rating_num', axis=1)
Y_t = train_set['rating_num']
Xt_cols = X_t.columns


from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


sel = SelectFromModel(RandomForestRegressor())
sel.fit(X_t, Y_t)

sel.get_support()

selected_feat= Xt_cols[(sel.get_support())]
len(selected_feat)


X_t = X_t[selected_feat]

Xt_cols = X_t.columns





corr_matrix = X_t.corr().abs()

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.99)]

print(to_drop)

X = X_t.drop(to_drop, axis=1)
Y = train_set['rating_num']

X_cols = X.columns







# Step 6:
# Splitting the datasets for model evaluation

from sklearn.model_selection import train_test_split

# Using 70% of data for training and 30 % for testing and stratifying on Y
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)


# Step 7:
# Scaling the data before applying the model
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)
X = mms.fit_transform(X)





#################################### PART C (Model Selection)###################################


# Since this is classification problem the following classification models give best results
    # RandomForest
    # XGBoost
    # LightGBM
    # CatBoost (works best if more categorical features are present: can be dropped for this problem)

# Importing all models as mentioned above

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor





# Calling the models
lr = LinearRegression()
rfc = RandomForestRegressor()
lgb = LGBMRegressor()
cb = CatBoostRegressor()
xgb = XGBRegressor()







# Creating Params for hyperparameter tuning


lgb_params = {
                'max_depth':[4,6,8],
                'num_leaves':[16, 64, 256],
                'learning_rate':[0.01, 0.04, 0.1],
                'num_iterations':[100, 1000, 2000, 4000, 6000],
                'feature_fraction':[0.6, 0.8, 1]
             }




# Step 2:
# Performing GridSearchCV for selecting best model

# Creating scorer for GridSearchCV
from sklearn.metrics import r2_score, make_scorer
f1_scorer = make_scorer(r2_score)

# Importing GridSearchCV from sklearn library
from sklearn.model_selection import GridSearchCV



# LGBM GridSearch
gsc_lgb = GridSearchCV(estimator=lgb, param_grid=lgb_params, cv=4, scoring=f1_scorer)
gsc_fit_lgb = gsc_lgb.fit(X, Y)
cv_results_lgb = pd.DataFrame.from_dict(gsc_fit_lgb.cv_results_)














lr.fit(x_train, y_train)

y_predict_lr = lr.predict(x_test)

score_lr = lr.score(x_test, y_test)


from sklearn.metrics import r2_score

r_score_lr = r2_score(y_test, y_predict_lr)






rfc.fit(x_train, y_train)

y_predict_rfc = rfc.predict(x_test)

score_rfc = rfc.score(x_test, y_test)

r_score_rfc = r2_score(y_test, y_predict_rfc)




cb.fit(x_train, y_train)

y_predict_cb = cb.predict(x_test)

score_cb = cb.score(x_test, y_test)

fi_cb = cb.feature_importances_

r_score_cb = r2_score(y_test, y_predict_cb)




lgb.fit(x_train, y_train)

y_predict_lgb = lgb.predict(x_test)

score_lgb = lgb.score(x_test, y_test)

r_score_lgb = r2_score(y_test, y_predict_lgb)



xgb.fit(x_train, y_train)

y_predict_xgb = xgb.predict(x_test)

score_xgb = xgb.score(x_test, y_test)

r_score_xgb = r2_score(y_test, y_predict_xgb)






################################# PART D (Running Model on Test Data)####################################


# Step 1:
# Running the model to predict results for test dataset

# Calling the model with parameters
lgb_main = LGBMRegressor()
lgb_main.fit(X, Y)


# Preparing the test set with selected features

test_set = test_set[selected_feat]
test_set = test_set.drop(to_drop, axis=1)

# Scaling the test set
test_set = mms.transform(test_set)

# Predicitng the results
test_predict_lgb = lgb_main.predict(test_set)


# Step 2:
# Exporting the Predicted Results

# Gathering ids from sample submision data
sample_data = pd.read_csv('sample_submission.csv')
sample_data = sample_data.drop('rating_num', axis=1)

# adding ids to the predicted results
test_results_lgb = pd.DataFrame(test_predict_lgb).rename(columns={0: 'rating_num'})
final_results_lgb = pd.concat([sample_data, test_results_lgb], axis=1)

# Exporting the final results for submission
export = final_results_lgb.to_csv('fball_hthon_09.csv', index=False)











################################# PART D (Running Model on Test Data)####################################


# Step 1:
# Running the model to predict results for test dataset

# Calling the model with parameters
cb_main = CatBoostRegressor()
cb_main.fit(X, Y)


# Preparing the test set with selected features

test_set = test_set[selected_feat]
test_set = test_set.drop(to_drop, axis=1)

# Scaling the test set
test_set = mms.transform(test_set)

# Predicitng the results
test_predict_cb = cb_main.predict(test_set)


# Step 2:
# Exporting the Predicted Results

# Gathering ids from sample submision data
sample_data = pd.read_csv('sample_submission.csv')
sample_data = sample_data.drop('rating_num', axis=1)

# adding ids to the predicted results
test_results_cb = pd.DataFrame(test_predict_cb).rename(columns={0: 'rating_num'})
final_results_cb = pd.concat([sample_data, test_results_cb], axis=1)

# Exporting the final results for submission
export = final_results_cb.to_csv('fball_hthon_06.csv', index=False)







################################# PART D (Running Model on Test Data)####################################


# Step 1:
# Running the model to predict results for test dataset

# Calling the model with parameters
xgb_main = XGBRegressor()
xgb_main.fit(X, Y)

# Preparing the test set with selected features
test_set = test_set[selected_feat]
test_set = test_set.drop(to_drop, axis=1)

# Scaling the test set
test_set = mms.transform(test_set)

# Predicitng the results
test_predict_xgb = xgb_main.predict(test_set)


# Step 2:
# Exporting the Predicted Results

# Gathering ids from sample submision data
sample_data = pd.read_csv('sample_submission.csv')
sample_data = sample_data.drop('rating_num', axis=1)

# adding ids to the predicted results
test_results_xgb = pd.DataFrame(test_predict_xgb).rename(columns={0: 'rating_num'})
final_results_xgb = pd.concat([sample_data, test_results_xgb], axis=1)

# Exporting the final results for submission
export = final_results_xgb.to_csv('fball_hthon_05.csv', index=False)







################################# PART D (Running Model on Test Data)####################################


# Step 1:
# Running the model to predict results for test dataset

# Calling the model with parameters
rfc_main = RandomForestRegressor()
rfc_main.fit(X, Y)

# Preparing the test set with selected features
test_set = test_set[selected_feat]
test_set = test_set.drop(to_drop, axis=1)

# Scaling the test set
test_set = mms.transform(test_set)

# Predicitng the results
test_predict_rfc= rfc_main.predict(test_set)


# Step 2:
# Exporting the Predicted Results

# Gathering ids from sample submision data
sample_data = pd.read_csv('sample_submission.csv')
sample_data = sample_data.drop('rating_num', axis=1)

# adding ids to the predicted results
test_results_rfc = pd.DataFrame(test_predict_rfc).rename(columns={0: 'rating_num'})
final_results_rfc = pd.concat([sample_data, test_results_rfc], axis=1)

# Exporting the final results for submission
export = final_results_rfc.to_csv('fball_hthon_04.csv', index=False)















