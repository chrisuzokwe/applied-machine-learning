import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

"""Polynomial Basis Functions: 
        uses:  target vector t
               input data x
               basis vector phi(1,x,x^2...x^M)
               weight vector w
               (technically) mapping function y(phi, w) <-- Matrix Multiplication
               cost function J(w) = sum(t-y(phi,w))^2

        find w with lowest cost J(w) by finding w* from deriv J(w) = 0"""

# Read data from csv using pandas
data = ["x","t"]
df = pd.read_csv('lsdata2.csv', names = data)
x = df["x"].to_numpy().reshape(200,1)
t = df["t"].to_numpy().reshape(200,1)
M = 2

# return our basis vector from x
def basis(M, x):
    i = 1
    phi = np.ones((len(x), 1))
    while i < M:
        phi = np.hstack((phi, x**(i)))
        i = i+1

    return phi

# calculate y from our basis vector and weights using normal equations
phi = basis(M, x)
w  = ((inv(np.dot(phi.transpose(), phi)))) @ phi.transpose() @ t
y = w.transpose() @ phi.transpose()

# find sorted data solution to plot best fit line
xsort = np.sort(x, axis = 0)
phisort = basis(M, xsort)
ysort = w.transpose() @ phisort.transpose()


# Calculate mean squared error
MSE = 0
for i in range(len(t)):
    MSE = MSE + (t[i] - y.transpose()[i])**2

MSE = MSE/len(t)
print(MSE)

# plot & MSE en masse
fig, ax = plt.subplots(nrows =4,ncols=2)

i = 0
for row in ax:
    for col in row:
        i = i+1
        if i == 8:
            break
        phi = basis(i*2, x)
        w = ((inv(np.dot(phi.transpose(), phi)))) @ phi.transpose() @ t
        y = w.transpose() @ phi.transpose()

        xsort = np.sort(x, axis=0)
        phisort = basis(i*2, xsort)
        ysort = w.transpose() @ phisort.transpose()

        axeslabel = "M =" + str(i*2)
        col.scatter(x, t, c='b', marker='o', label='Data')
        col.plot(xsort, ysort.transpose(), 'r', label=axeslabel)

        MSE = 0
        for j in range(len(t)):
            MSE = MSE + (t[j] - y.transpose()[j]) ** 2

        MSE = MSE / len(t)
        print(MSE)

        col.legend()

plt.show()


"""# plot data & best fit line
plt.figure(1)
plt.scatter(x,t, c='b',marker='o', label='Data')
plt.plot(xsort,ysort.transpose(), 'r', label = 'Best Fit Line | M ')

plt.xlabel('x')
plt.ylabel('t')
plt.title('Polynomial Basis Functions')
plt.legend()
plt.show()"""


