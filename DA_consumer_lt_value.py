

import pandas as pd
import numpy as np
import sklearn


data = pd.read_csv("Insurance_Marketing-Customer-Value-Analysis.csv",usecols = ['Customer Lifetime Value','Coverage','Income','Monthly Premium Auto','Number of Open Complaints','Number of Policies'])

data.insert(6,"Basic",0)
data.insert(7,"Premium",0)
data.insert(8,"Extended",0)

for i in range(0,len(data)):
  if data.loc[i,'Coverage'] == 'Basic' :
    data.loc[i,"Basic"] = 1
  elif data.loc[i,'Coverage'] == 'Premium' :
    data.loc[i,"Premium"] = 1
  elif data.loc[i,'Coverage'] == 'Extended' :
    data.loc[i,"Extended"] = 1

data2 = data.to_numpy()
y = data2[:,0]
X = data2[:,2:8]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 10 )

from sklearn.preprocessing import StandardScaler

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# #sc_y = StandardScaler()
# #y_train = sc_y.fit_transform(y_train.reshape(-1,1))

n_y_test = np.reshape(y_test, (-1, y_test.shape[0]))
Xy_test = np.concatenate((X_test,n_y_test.T),axis = 1)
n_y_train = np.reshape(y_train, (-1, y_train.shape[0]))
Xy_train = np.concatenate((X_train,n_y_train.T),axis = 1)


import matplotlib.pyplot as plt
from pandas import DataFrame

import datetime
import re 

perc =[.20, .40, .60, .80] 
include =['object', 'float', 'int']

arr = dict()
j = 0
for i in Xy_train.T:
  arr[j] = i
  j += 1
abc = pd.DataFrame(arr)

#desc = abc.describe(percentiles = perc, include = include) 
#print(desc)
abc.corr(method = 'pearson')
#desc = abc.describe(percentiles = perc, include = include) 
#print(desc)

data4 = abc.to_numpy()
y_train = data4[:,6]
X_train = data4[:,0:6]
# y_fin_2 = np.reshape(y_fin, (-1, y_fin.shape[0]))
# y_fin = y_fin_2
# X_fin = pd.DataFrame(X_fin)
# y_fin = pd.DataFrame(y_fin)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

X12 = X_train.astype('float')
y12 = y_train.astype('float')

Y12=y12.astype(int)
y_test2 = y_test.astype(int)

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()


model.fit(X12,Y12)

y_pred = model.predict(X_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))