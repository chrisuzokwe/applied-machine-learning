import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
"""
Principal Component Analysis
    uses: [n-samples] of [D-dimensional Data]
          Projection of [data vector xk] onto [data mean (m)] line
          Coefficient to describe projection [ak]
          Unit vector direction [e]
          Scatter matrix S
          
    goal: reduce the dimensionality of the data i.e. find the [ak] coefficients that project us onto our [e] lines
"""

# load data using pandas
df = pd.read_csv('MNIST2class.csv', header=None).to_numpy()

# view a single image
img = df[0][1::].reshape(28,28)
#plt.imshow(img)
#plt.show()

# find the data mean
m = np.zeros((1, len(img.flatten())))
for row in df:
    m = m + row[1::]
m = m/2000

# view mean image
#m_img = m.reshape(28,28)
#plt.imshow(m_img)
#plt.show()

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

# view a principal component vector
img = EVs[0][1].copy().reshape(28,28)
#plt.imshow(img.real)
#plt.title('lambda = ' + str(EVs[0][0].real))
#plt.show()

# two-dimensional projection
transformer = np.array([EVs[4][1], EVs[3][1]])
print(EVs[4][1])
projection = []

for row in df[:, 1:]:
    aks = (row - m) @ transformer.T
    projection.append(aks)

projection = np.asarray(projection)
plt.scatter(projection[0:1000,0,0].real,projection[0:1000,0,1].real, c='y',marker='o', label='Digit 4')
plt.scatter(projection[1001:2000,0,0].real,projection[1001:2000,0,1].real, c='b',marker='o', label='Digit 8')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('PCA decomposition')
plt.legend()
plt.show()