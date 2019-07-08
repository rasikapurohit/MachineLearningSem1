#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

np.set_printoptions(threshold=np.nan, suppress=True)
# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]
# modified the features
X_train_mod = np.insert(X_train, -1, 1/X_train[:, 0], axis=1)
X_train_mod = np.insert(X_train_mod, -1, X_train[:, 5]**2,axis=1)
X_train_mod = X_train_mod[:,1:]

X_test_mod = np.insert(X_test, -1, 1/X_test[:, 0], axis=1)
X_test_mod = np.insert(X_test_mod, -1,X_test[:,5]**2, axis=1)
X_test_mod = X_test_mod[:,1:]

X_train_CV,X_test_CV,y_train_CV,y_test_CV = train_test_split(X_train_mod, y_train, test_size=0.1,random_state=2018)

param_grid = {
            'bootstrap': [True],
            'max_depth': [70,80, 90, 100, 110, 120,130],
            'max_features': [2,3,4],
            'min_samples_leaf': [2,3, 4, 5],
            'min_samples_split': [2,3,4,5,6,],
            'n_estimators': [200,300,400,500,800,1000]
            }
# use randomForestRegressor to train data
cpu_model = RandomForestRegressor(bootstrap=True, max_depth=70, max_features=4, min_samples_leaf=2, min_samples_split=2, n_estimators=300)
# use Grid search to estimate parameter
#grid_search = GridSearchCV(estimator = cpu_model, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
# fit the model
cpu_model.fit(X_train_CV, y_train_CV)
#grid_search.fit(X_train_CV, y_train_CV)
#print(grid_search.best_params_)
# predict the target value for test set
y_train_pred = cpu_model.predict(X_test_CV)
y_train_pred = np.maximum(y_train_pred, 0)
# calculate accuracy between the predicted and the actual value
print("Accuracy:",metrics.r2_score(y_train_pred, y_test_CV))


# Fit model and predict test values
y_pred = cpu_model.predict(X_test_mod)
y_pred = np.maximum(y_pred, 0)


# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,PRP"
n_points = X_test_mod.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

