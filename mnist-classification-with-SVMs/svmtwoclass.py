from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
Support Vector Machines -  Two Class Data
"""

# load dataframes using pandas
train = pd.read_csv('TwoClassTrain.csv', header=None)
test = pd.read_csv('TwoClassTrain.csv', header=None)

# split dataframes into labels & data
traindata = train[train.columns[1:]].to_numpy()
trainclasses = train[train.columns[0:1]].to_numpy()

testdata = test[test.columns[1:]].to_numpy()
testclasses = test[test.columns[0:1]].to_numpy()

# find the linear fit & prediction accuracy
lin = svm.SVC(kernel='linear')
lin.fit(traindata, trainclasses.ravel())
linpredictions = lin.predict(testdata)
linacc = accuracy_score(testclasses, linpredictions)
print(linacc)

# find the poly fit & prediction accuracy
poly = svm.SVC(kernel='poly', degree=4, C=56)
poly.fit(traindata, trainclasses.ravel())
polypredictions = poly.predict(testdata)
polyacc = accuracy_score(testclasses, polypredictions)
print(polyacc)

# find the rbf fit & prediction accuracy
rbf = svm.SVC(kernel='rbf')
rbf.fit(traindata, trainclasses.ravel())
rbfpredictions = rbf.predict(testdata)
rbfacc = accuracy_score(testclasses, rbfpredictions)
print(rbfacc)


