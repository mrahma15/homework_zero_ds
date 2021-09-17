#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[89]:


df = pd.read_csv('https://raw.githubusercontent.com/mrahma15/homework_zero_ds/main/D3.csv')
X1 = df.values[:, 0]
X2 = df.values[:, 1]
X3 = df.values[:, 2]
y = df.values[:, 3]
m = len(y)
print('X1 = ', X1[: 5]) 
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])
print('y = ', y[: 5])
print('m = ', m)


# In[90]:


plt.scatter(X1,y, color='red',marker= '+')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Variable 1')
plt.ylabel('Output')
plt.title('Scatter plot of training data')


# In[91]:


plt.scatter(X2,y, color='red',marker= '+')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Variable 2')
plt.ylabel('Output')
plt.title('Scatter plot of training data')


# In[92]:


plt.scatter(X3,y, color='red',marker= '+')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Variable 3')
plt.ylabel('Output')
plt.title('Scatter plot of training data')


# In[93]:


X_0 = np.ones((m, 1))
X_0[:5]


# In[94]:


X_1 = X1.reshape(m, 1)
X_2 = X2.reshape(m, 1)
X_3 = X3.reshape(m, 1)
print('X_1 = ', X_1[: 5])
print('X_2 = ', X_2[: 5])
print('X_3 = ', X_3[: 5])


# In[95]:


X1 = np.hstack((X_0, X_1))
X2 = np.hstack((X_0, X_2))
X3 = np.hstack((X_0, X_3))
print('X1 = ', X1[: 5])
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])


# In[96]:


theta = np.zeros(2)
theta


# In[97]:


def compute_cost(A, y, theta):
    predictions = A.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J


# In[98]:


def gradient_descent(A, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = A.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * A.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compute_cost(A, y, theta)
    
    return theta, cost_history


# In[99]:


cost_1 = compute_cost(X1, y, theta)
cost_2 = compute_cost(X2, y, theta)
cost_3 = compute_cost(X3, y, theta)
print('The cost for given values of theta and Variable 1 =', cost_1)
print('The cost for given values of theta and Variable 2 =', cost_2)
print('The cost for given values of theta and Variable 3 =', cost_3)


# In[100]:


theta = [0., 0.]
iterations = 1500;
alpha = 0.01


# In[101]:


theta, cost_history = gradient_descent(X1, y, theta, alpha, iterations)
print('Final value of theta (Variable 1)=', theta)
print('cost_history (Variable 1) =', cost_history)


# In[102]:


plt.scatter(X1[:,1], y, color='red', marker= '+', label= 'Training Data')
plt.plot(X1[:,1],X1.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Variable 1')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[103]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[104]:


theta, cost_history = gradient_descent(X2, y, theta, alpha, iterations)
print('Final value of theta (Variable 2)=', theta)
print('cost_history (Variable 2) =', cost_history)


# In[105]:


plt.scatter(X2[:,1], y, color='red', marker= '+', label= 'Training Data')
plt.plot(X2[:,1],X2.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Variable 2')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[106]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[107]:


theta, cost_history = gradient_descent(X3, y, theta, alpha, iterations)
print('Final value of theta (Variable 3)=', theta)
print('cost_history (Variable 3) =', cost_history)


# In[108]:


plt.scatter(X3[:,1], y, color='red', marker= '+', label= 'Training Data')
plt.plot(X3[:,1],X3.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Variable 3')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[87]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')

