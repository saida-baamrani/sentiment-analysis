# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 00:42:32 2019

@author: saida
"""
from cleaner import TwitterCleanuper 
from TokenStem import TokenizationStemming
from Naitive_Biais import Naive_Bais
from RandomForest import Random_Forest
from SupportVectorMachine import Support_Vector
from BagOfWords import Bag_of_words
from AccuracyPerformence import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.tokenize import word_tokenize
import csv
from sklearn.model_selection import train_test_split 

df= pd.read_csv("./data set/train_data.xls",quoting=csv.QUOTE_NONE,
                sep='\t',header=None,names=['review','opinion'])


tk= TokenizationStemming()

#data=tk.stem(data)
#print(data.head())
#data1=tk.tokenize(df)
#data1=tk.stem(data1)

#print(data1)
cl= TwitterCleanuper()
for cleanup_method in cl.iterate():
    data = cleanup_method(df)
    #data=df.cl.iterate()
data=tk.stemming(data)

data=Bag_of_words(data) 
y=df.review
X=data

train_X, val_X, train_y, val_y = train_test_split(X, y,test_size = 0.3, random_state = 0,shuffle=True)

print('*******Naive Bais*****')
model_naive=Naive_Bais(train_X, train_y)

train_predict=predict(model_naive,val_X)
proba=predict_proba(model_naive,val_X)

performance(val_y,proba)

mean_abs_err(train_predict,val_y)



print('*******Random Forest*****')

model_forest=Random_Forest(train_X, train_y)

train_predict=predict(model_forest,val_X)
proba=predict_proba(model_forest,val_X)
performance(val_y,proba)

mean_abs_err(train_predict,val_y)

'''print('*******XGBOOST *****')

model_xgboost=XGBoost_classier(train_X, train_y)

train_predict=predict(model_xgboost,val_X)
proba=predict_proba(model_xgboost,val_X)
performance(val_y,proba)

mean_abs_err(train_predict,val_y)'''

print('*******SVM *****')

model_SVM=Support_Vector(train_X, train_y)

train_predict=predict(model_SVM,val_X)
proba=predict_proba(model_SVM,val_X)
performance(val_y,proba)

mean_abs_err(train_predict,val_y)







   

