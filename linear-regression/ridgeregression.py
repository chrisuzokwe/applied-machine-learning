import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

"""Ridge Regression: 
        uses:  target vector t
               input data x
               basis vector phi(1,x,x^2...x^M)
               weight vector w
               (technically) mapping function y(phi, w) <-- Matrix Multiplication
               cost function J(w) = sum(t-y(phi,w))^2 + lambda/2 *wTw

        find w with lowest cost J(w) by finding w* from deriv J(w) = 0"""

# Read data from csv using pandas
data = ["x","t"]
df = pd.read_csv('lsdata2.csv', names = data)
x = df["x"].to_numpy().reshape(200,1)
t = df["t"].to_numpy().reshape(200,1)
l = 10**(-14)
M = 12

# return our basis vector from x
def basis(M, x):
    i = 1
    phi = np.ones((len(x), 1))
    while i < M:
        phi = np.hstack((phi, x**(i)))
        i = i+1

    return phi


phi = basis(M, x)
xsort = np.sort(x, axis = 0)
phisort = basis(M, xsort)


# ridge regression solution
reg = l*np.identity(M)
w  = ((inv(reg+np.dot(phi.transpose(), phi)))) @ phi.transpose() @ t
yridge = w.transpose() @ phisort.transpose()
yridgetest = w.transpose() @ phi.transpose()

# polynomial w/o regularization
w  = ((inv(np.dot(phi.transpose(), phi)))) @ phi.transpose() @ t
yreg = w.transpose() @ phisort.transpose()
yregtest = w.transpose() @ phi.transpose()

# plot information
plt.figure(1)
plt.plot(xsort,yreg.transpose(), 'g.-', label = 'Regular Fit Line', linewidth=2)
plt.plot(xsort,yridge.transpose(), 'r--', label = 'Ridge Regression Fit | l = 10^-14')
plt.scatter(x,t, c='b',marker='o', label='Data')

plt.xlabel('x')
plt.ylabel('t')
plt.title('Ridge Regression vs. Polynomial Basis Functions | M = 12')
plt.legend()
plt.show()

#MSE Calculation
MSE = 0
for i in range(len(t)):
    MSE = MSE + (t[i] - yregtest.transpose()[i])**2

MSE = MSE/len(t)
print("Basis funcion MSE:", MSE)

MSE = 0
for i in range(len(t)):
    MSE = MSE + (t[i] - yridgetest.transpose()[i])**2

MSE = MSE/len(t)
print("Ridge funcion MSE:", MSE)