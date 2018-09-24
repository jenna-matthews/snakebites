#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 17:16:59 2018

@author: jennaolsen
Capgemini Take Home Test - prediction of 911 call reason using longitude and latitude
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

df_911 = pd.read_excel("/Users/jennaolsen/Downloads/test911.xlsx")

df_911.head()


#Drop the unnecessary column (not used for prediction)
df_911.drop(df_911.columns[3], axis=1, inplace=True)



#Split data into training & test sets
X = df_911.iloc[:,1:]
y = df_911.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#Checking the training & test sets
X_train.shape
y_train.head()


###Logistic Regression first - gives a starting point for predictions
#Fit Logistic Regression to the data set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Check accuracy of predictions at first pass
y_pred = classifier.predict(X_test)
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

""" Results
[[94 34  0  0]   - 94 accurate, 34 inaccurate
 [43 54  0  0]   - 54 accurate, 43 inaccurate
 [67  6  3  1]   - 3 accurate, 74 inaccurate
 [76  0  1  0]]  - 0 accurate, 77 inaccurate
Accuracy of logistic regression classifier on test set: 0.40
"""

from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
#cross validation
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
"""
10-fold cross validation average accuracy: 0.446
"""

#K-nearest neighbors option
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train) 

y_pred_knn = knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred_knn)
"""
Accuracy Score = 0.97361477572559363 -- very satisfactory
"""
cm = confusion_matrix(y_test.tolist(), y_pred_knn.tolist())
print(cm)
""" Results -10 total incorrect; 369 total correct
[[128   0   0   0]
 [  0  97   0   0]
 [  0   7  70   0]
 [  1   1   1  74]]
"""