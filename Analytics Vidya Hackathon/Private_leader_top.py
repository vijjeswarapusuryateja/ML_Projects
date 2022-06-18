# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:15:49 2022

@author: CHITTIBABU
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')


data = pd.concat([train_data, test_data], axis=0)

cols = data.columns

data_types = data.dtypes

missing_vals = data.isnull().sum(axis=0)

df = data.copy()



df['signup_date'] = pd.to_datetime(df['signup_date'])

df['created_at'] = pd.to_datetime(df['created_at'])

df['user_signup'] = df['signup_date']



df['signup_date'] = df['signup_date'].fillna(df['created_at'])



df['signup_day'] = df['signup_date'].dt.day

df['signup_month'] = df['signup_date'].dt.month

df['signup_year'] = df['signup_date'].dt.year




#NEW COLUMNS
df['signup_activity'] = df['user_signup'].dt.year

df['user_activity_var_0'] = df['signup_activity']

df['user_activity_var_0'] = df['user_activity_var_0'].fillna(0)

df.loc[df['user_activity_var_0'] > 2000, 'user_activity_var_0'] = 1



df['total_campaigns'] = df['campaign_var_1'] + df['campaign_var_2']

df['total_activity'] = df.iloc[:, 6:18].sum(axis=1)

df['total_activity'] = df['total_activity'] + df['user_activity_var_0']

df = df.drop(['id', 'created_at', 'signup_date', 'products_purchased', 'campaign_var_1', 'campaign_var_2',
              'user_activity_var_3', 'user_activity_var_6', 'signup_day', 
              'user_signup', 'signup_activity'], axis=1)




corr_matrix = df.corr()




features_set = df.copy()

train_set = features_set.iloc[:len(train_data), :]

test_set = features_set.iloc[len(train_data):, :]



X = train_set.drop('buy', axis=1)

X_t = train_set.drop('buy', axis=1)

Y = train_set['buy']

X_cols = X.columns



from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

X = mms.fit_transform(X)



from sklearn.feature_selection import SelectKBest, chi2

kbest = SelectKBest(chi2, k='all')
kbest.fit_transform(X, Y)
pvalues = kbest.pvalues_



from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1234, stratify=Y)

x_train, x_test, y_train, y_test = train_test_split(X_t, Y, test_size=0.1, random_state=5678, stratify=Y)



mms2 = MinMaxScaler()

x_train = mms2.fit_transform(x_train)

x_test = mms2.transform(x_test)





from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=4, objective='binary:logistic', learning_rate=0.01, n_estimators=2000, colsample_bytree=0.7)

xgb_main = XGBClassifier(max_depth=4, objective='binary:logistic', learning_rate=0.01, n_estimators=2000, colsample_bytree=0.7)

#xgb.fit(x_train, y_train)

xgb_main.fit(X, Y)

y_predict_xgb = xgb.predict(x_test)

y_proba_xgb = xgb.predict_proba(x_test)

score_xgb = xgb.score(x_test, y_test)

fi_xgb = xgb.feature_importances_


from sklearn.metrics import confusion_matrix, f1_score

cm_xgb = confusion_matrix(y_test, y_predict_xgb)

f1_xgb = f1_score(y_test, y_predict_xgb)





from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc_main = RandomForestClassifier()

rfc.fit(x_train, y_train)

rfc_main.fit(X, Y)

y_predict_rfc = rfc.predict(x_test)

y_proba_rfc = rfc.predict_proba(x_test)

score_rfc = rfc.score(x_test, y_test)

fi_rfc = rfc.feature_importances_




cm_rfc = confusion_matrix(y_test, y_predict_rfc)

f1_rfc = f1_score(y_test, y_predict_rfc)




from catboost import CatBoostClassifier

cb = CatBoostClassifier(depth=6, learning_rate=0.01, iterations=2000, rsm=0.7)

cb_main = CatBoostClassifier(depth=6, learning_rate=0.01, iterations=2000, rsm=0.7)

cb.fit(x_train, y_train)

cb_main.fit(X, Y)

y_predict_cb = cb.predict(x_test)

y_proba_cb = cb.predict_proba(x_test)

score_cb = cb.score(x_test, y_test)

fi_cb = cb.feature_importances_


cm_cb = confusion_matrix(y_test, y_predict_cb)

f1_cb = f1_score(y_test, y_predict_cb)




from lightgbm import LGBMClassifier

lgb = LGBMClassifier(max_depth=4, num_leaves=16, learning_rate=0.01, n_estimators=2000)

lgb_main = LGBMClassifier(max_depth=4, num_leaves=16, learning_rate=0.01, n_estimators=2000)

lgb.fit(x_train, y_train)

lgb_main.fit(X, Y)

y_predict_lgb = lgb.predict(x_test)

y_proba_lgb = lgb.predict_proba(x_test)

score_lgb = lgb.score(x_test, y_test)

fi_lgb = lgb.feature_importances_


cm_lgb = confusion_matrix(y_test, y_predict_lgb)

f1_lgb = f1_score(y_test, y_predict_lgb)








sample_data = pd.read_csv('sample_submission.csv')

sample_data = sample_data.drop('buy', axis=1)

test_set = test_set.drop('buy', axis=1)

test_set = mms.transform(test_set)

test_predict_lgb = lgb_main.predict(test_set)

test_results_lgb = pd.DataFrame(test_predict_lgb).rename(columns={0: 'buy'})

final_results_lgb = pd.concat([sample_data, test_results_lgb], axis=1)

export = final_results_lgb.to_csv('av_jthon_03.csv', index=False)







sample_data = pd.read_csv('sample_submission.csv')

sample_data = sample_data.drop('buy', axis=1)

test_set = test_set.drop('buy', axis=1)

test_set = mms.transform(test_set)

test_predict_rfc = rfc_main.predict(test_set)

test_results_rfc = pd.DataFrame(test_predict_rfc).rename(columns={0: 'buy'})

final_results_rfc= pd.concat([sample_data, test_results_rfc], axis=1)

export = final_results_rfc.to_csv('av_jthon_04.csv', index=False)






sample_data = pd.read_csv('sample_submission.csv')

sample_data = sample_data.drop('buy', axis=1)

test_set = test_set.drop('buy', axis=1)

test_set = mms.transform(test_set)

test_predict_xgb = xgb_main.predict(test_set)

test_results_xgb = pd.DataFrame(test_predict_xgb).rename(columns={0: 'buy'})

final_results_xgb= pd.concat([sample_data, test_results_xgb], axis=1)

export = final_results_xgb.to_csv('av_jthon_05.csv', index=False)











































