import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
Fisher's Linear Discriminant
    uses: Datasets [#1] and [#2]
          N total samples of data
                    
    goal: find w that creates y1 and y2 -- to cluster the data while projecting it into a new space
"""

# read data from arrays, create dataset and crossval split
C1 = pd.read_csv('fldC1.csv', header=None).to_numpy()
C2 = pd.read_csv('fldC2.csv', header=None).to_numpy()

label1 = np.ones((len(C1), 1))
label2 = np.ones((len(C2), 1))*2

C1 = np.concatenate((label1, C1), axis=1)
C2 = np.concatenate((label2, C2), axis=1)

set = 5
split1 = np.split(C1, 5)
split2 = np.split(C2, 5)

Val = []
C1 = []
C2 = []

for i in range(len(split1)):
    if i+1 == set:
        Val = np.concatenate((split1[i], split2[i]), axis=0)
    else:
        C1.extend(split1[i])
        C2.extend(split2[i])

D = np.concatenate((C1, C2), axis=0)

C1 = np.asarray(C1)
C2 = np.asarray(C2)

# calculate means
m1 = np.mean(C1[:,[1,2]], axis=0)
m2 = np.mean(C2[:,[1,2]], axis=0)

N1 = len(C1)
N2 = len(C2)
N = len(D)


# calculate within-class scatter matrix
## scatter matrix 1
S1 = np.zeros((2, 2))
for row in C1:
    accum = ((row[1::] - m1).T * (row[1::] - m1))
    S1 = S1 + accum

## scatter matrix 2
S2 = np.zeros((2, 2))
for row in C2:
    accum = ((row[1::] - m2).T * (row[1::] - m2))
    S2 = S2 + accum

Sw = S1 + S2

# calculate w
w = np.linalg.pinv(Sw) @ (m1 - m2)

# calculate data mean
m = (N1*m1 + N2*m2)/N

# calculate accuracy across validation set
correct = 0
for row in Val:
    if w.T @ (row[1::]-m) > 0:
        c = 1
    else:
        c = 2

    if c == row[0]:
        correct = correct + 1


accuracy = correct/len(Val)
print(accuracy)


# project in the 1-D space
y = []

for row in Val:
    y.append([row[0], w.T @ (row[1::]-m)])

y = np.asarray(y)
#plt.hist(y[0:200, [1]], bins=100, label="Class 1",  edgecolor='black', linewidth=0.5)
#plt.hist(y[200:400, [1]], bins=100, label="Class 2", edgecolor='black', linewidth=0.5)
#plt.xlabel("y")
#plt.ylabel("count")
#plt.title('Fisher\'s Linear Discriminant Seperation')
#plt.legend()
#plt.show()


