import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Linear Regression: 
        uses:  target vector t
               weight vector w
               input data x
               (technically) mapping function y(x, w) <-- Matrix Multiplication
               cost function J(w) = sum(t-y(x,w))^2
               
        find w with lowest cost J(w) by finding w* from deriv J(w) = 0"""

# Read data from csv using pandas
data = ["x","t"]
df = pd.read_csv('lsdata1.csv', names = data)
x = df["x"].to_numpy().reshape(20,1)
t = df["t"].to_numpy().reshape(20,1)

# Calculate w* normal equations, compute y(x,w)
w = ((1/(np.dot(x.transpose(), x)))) @ x.transpose() @ t
y = w*x
print("parameter vector -- ", w)

# Plot figure
plt.figure(1)
plt.plot(x,y, 'r', label = 'Best Fit Line')
plt.scatter(x,t, c='b',marker='o', label='Data')

plt.xlabel('x')
plt.ylabel('t')
plt.title('Linear Regression')
plt.legend()
plt.show()




