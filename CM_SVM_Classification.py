#!/usr/bin/env python

import numpy as np
from sklearn import neighbors
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split


# load all the files
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:, 1]
X_test_final = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)

X_train_mod=np.array(X_train)
X_test_mod=np.array(X_test_final)

# split the data
X_train_CV,X_test_CV,y_train_CV,y_test_CV = train_test_split(X_train_mod, y_train, test_size=0.2,random_state=2018)

# preprocessed data using quantile transformer
scaler = preprocessing.QuantileTransformer(random_state=0).fit(X_train_mod)
X_train_preprocess = scaler.transform(X_train_CV)
X_test_preprocess = scaler.transform(X_test_CV)
X_test_final_preprocess = scaler.transform(X_test_mod)

# support vector machine classification using linear kernel
clf = svm.SVC(C=1.0,kernel='linear',decision_function_shape='ovr')
clf.fit(X_train_preprocess,y_train_CV)

# predict data for 2nd part
y_pred_test = clf.predict(X_test_preprocess)
# compare the accuracy for predicted target and actual target
print("Accuracy:",metrics.accuracy_score(y_pred_test, y_test_CV))

# predict data for given test input
y_pred = clf.predict(X_test_final_preprocess)


# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,EpiOrStroma"
n_points = X_test_mod.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

