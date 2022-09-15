# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import nltk
import string
import re

nltk.download('stopwords')
nltk.download('wordnet')


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

data = pd.concat([train_data, test_data], axis=0)

df = data.copy()

df['label'].value_counts()


# Removing URLs
def remove_url(text):
    return re.sub(r"http\S+", "", text)

#Removing Punctuations
def remove_punct(text):
    new_text = []
    for t in text:
        if t not in string.punctuation:
            new_text.append(t)
    return ''.join(new_text)


#Tokenizer
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')



#Removing Stop words
from nltk.corpus import stopwords

def remove_sw(text):
    new_text = []
    for t in text:
        if t not in stopwords.words('english'):
            new_text.append(t)
    return new_text

#Lemmatizaion
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def word_lemmatizer(text):
    new_text = []
    for t in text:
        lem_text = lemmatizer.lemmatize(t)
        new_text.append(lem_text)
    return new_text
        


        


df['tweet'] = df['tweet'].apply(lambda t: remove_url(t))

df['tweet'] = df['tweet'].apply(lambda t: remove_punct(t))

df['tweet'] = df['tweet'].apply(lambda t: tokenizer.tokenize(t.lower()))

df['tweet'] = df['tweet'].apply(lambda t: remove_sw(t))

df['tweet'] = df['tweet'].apply(lambda t: word_lemmatizer(t))



features_set = df.copy()

train_set = features_set.iloc[:len(train_data), :]

test_set = features_set.iloc[len(train_data):, :]

X = train_set['tweet']


for i in range(0, len(X)):
    X.iloc[i] = ' '.join(X.iloc[i])


Y = train_set['label']


from sklearn.feature_extraction.text import TfidfVectorizer

TfidV = TfidfVectorizer()

X = TfidV.fit_transform(X)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1234)



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

y_predict_lr = lr.predict(x_test)

from sklearn.metrics import confusion_matrix, f1_score

cm_lr = confusion_matrix(y_test, y_predict_lr)

f1_lr = f1_score(y_test, y_predict_lr)

score_lr = lr.score(x_test, y_test)



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(class_weight='balanced')

rfc.fit(x_train, y_train)

y_predict_rfc = rfc.predict(x_test)

from sklearn.metrics import confusion_matrix, f1_score

cm_rfc = confusion_matrix(y_test, y_predict_rfc)

f1_rfc = f1_score(y_test, y_predict_rfc)

score_rfc = rfc.score(x_test, y_test)



from xgboost import XGBClassifier

xgb = XGBClassifier(scale_pos_weight=3)

xgb.fit(x_train, y_train)

y_predict_xgb = xgb.predict(x_test)

from sklearn.metrics import confusion_matrix, f1_score

cm_xgb = confusion_matrix(y_test, y_predict_xgb)

f1_xgb = f1_score(y_test, y_predict_xgb)

score_xgb = xgb.score(x_test, y_test)




from lightgbm import LGBMClassifier

lgb = LGBMClassifier(scale_pos_weight=3)

lgb.fit(X, Y)

#lgb.fit(x_train, y_train)

y_predict_lgb = lgb.predict(x_test)

from sklearn.metrics import confusion_matrix, f1_score

cm_lgb = confusion_matrix(y_test, y_predict_lgb)

f1_lgb = f1_score(y_test, y_predict_lgb)

score_lgb = lgb.score(x_test, y_test)





sample_submission = pd.read_csv('sample_submission.csv')

test_set = test_set.drop('label', axis=1)

test_X = test_set['tweet']

for i in range(0, len(test_X)):
    test_X.iloc[i] = ' '.join(test_X.iloc[i])

test_X = TfidV.transform(test_X)



test_predict = xgb.predict(test_X)

test_result = pd.DataFrame(test_predict).rename(columns={0:'label'})

final_result = pd.concat([sample_submission['id'], test_result], axis=1)

export = final_result.to_csv('sent_text_01.csv')




test_predict = lgb.predict(test_X)

test_result = pd.DataFrame(test_predict).rename(columns={0:'label'})

final_result = pd.concat([sample_submission['id'], test_result], axis=1)

export = final_result.to_csv('sent_text_03.csv', index=False)
