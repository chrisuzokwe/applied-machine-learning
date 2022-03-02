from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
"""
Support Vector Machines -  Two Class MNIST projection
"""

# load data using pandas
df = pd.read_csv('MNIST2class.csv', header=None).to_numpy()

# view a single image
img = df[0][1::].reshape(28,28)

# find the data mean
m = np.zeros((1, len(img.flatten())))
for row in df:
    m = m + row[1::]
m = m/2000

# calculate the scatter matrix
S = np.zeros((784, 784))
for row in df:
    accum = ((row[1::] - m).T * (row[1::] - m))
    S = S + accum

# find the eigenvalues and vectors for this Scatter Matrix
EVs = np.linalg.eig(S)

# remove eigenvectors who's magnitude is not 1 & sort by largest corresponding eigenvalues
EVtemp = []
for i in range(len(EVs[0])):
    if np.linalg.norm(EVs[1][:,i]) == 1:
        EVtemp.append([EVs[0][i].copy(), EVs[1][:,i].copy()])
EVtemp = np.asarray(EVtemp)
EVs = EVtemp[np.argsort(EVtemp[:, 0])][-5:]

# two-dimensional projection
transformer = np.array([EVs[4][1], EVs[3][1]])
projection = []

for row in df[:, 1:]:
    aks = (row - m) @ transformer.T
    projection.append(aks)
projection = np.asarray(projection)

# split projection into training data
traindatafour = np.vstack((projection[0:800,0,0].real, projection[0:800,0,1].real)).T
traindataeight = np.vstack((projection[1000:1800,0,0].real, projection[1000:1800,0,1].real)).T
traindata = np.vstack((traindatafour, traindataeight))
trainclasses = np.vstack((np.ones((800,1))*4, np.ones((800,1))*8))

# ...and test data
testdatafour = np.vstack((projection[800:1000,0,0].real, projection[800:1000,0,1].real)).T
testdataeight = np.vstack((projection[1800:2000,0,0].real, projection[1800:2000,0,1].real)).T
testdata = np.vstack((testdatafour, testdataeight))
testclasses = np.vstack((np.ones((200,1))*4, np.ones((200,1))*8))

# find the linear fit & prediction accuracy
lin = svm.SVC(kernel='linear')
lin.fit(traindata, trainclasses.ravel())
linpredictions = lin.predict(testdata)
linacc = accuracy_score(testclasses, linpredictions)
print(linacc)

# find the poly fit & prediction accuracy
poly = svm.SVC(kernel='poly', degree=4)
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