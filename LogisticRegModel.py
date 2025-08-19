#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegModel:
    def __init__(self, x_values, y_values,alpha, w_init = 0, b_init = 0):
        self.x = np.array(x_values)
        self.y = np.array(y_values)
        self.w = w_init
        self.b = b_init 
        self.alpha = alpha 
        self.m = len(y_values)
        self.cost_history = []
        self.w_history = []
        self.b_history = []

    def get_z(self):
        return self.w * self.x + self.b 

    def EstLogReg(self):
        return 1/(1+np.exp(-1*self.get_z()))

    def cost_fxn(self):
        loss= -1*self.y*np.log(self.EstLogReg())-(1-self.y)*(np.log(1-self.EstLogReg()))
        return np.sum(loss)*(1/self.m)

    def step_gradient_descent(self):
        z = self.get_z()
        y_hat =  self.EstLogReg()
        error = y_hat - self.y
        
        derivative_w = np.sum(error*self.x)*(1/self.m)
        w_new = self.w - self.alpha*derivative_w
    
        derivative_b = np.sum(error)*(1/self.m)
        b_new = self.b - self.alpha*derivative_b

        self.w ,self.b = w_new, b_new
        return w_new, b_new

    def train(self,iterations =10):
         for i in range(iterations):
             self.step_gradient_descent()
             self.w = round(self.w, 5)
             self.b = round(self.b, 5)
             cost_val = round (self.cost_fxn(), 5)
             self.cost_history.append(cost_val) 
             self.w_history.append(self.w)
             self.b_history.append(self.b)
         print(f"Iter {i+1}: w={self.w:.5f}, b={self.b:.5f}, cost_val={self.cost_fxn():.5f}")
         print(f"Logistic Regression Model: y = 1/(1+e^-({self.w:.5f}x +{self.b:.5f}))")
             
    def plot_cost(self):
        plt.plot(range(1, len(self.cost_history) + 1), self.cost_history)
        plt.xlabel("Iteration")
        plt.ylabel("Cost J(w, b)")
        plt.title("Cost function over iterations")
        plt.show()
            
def get_values(name):
    return list(map(float, input(f"Enter {name} values separated by commas: ").split(",")))
            
x_values = get_values("x")
y_values = get_values("y")
alpha  = float(input("Enter the learning rate:"))
iterations = int(input("Enter number of training iterations: "))

model = LogisticRegModel(
    x_values=x_values,
    y_values=y_values,
    alpha=alpha,
    w_init=0,
    b_init=0
)

model.train(iterations)
model.plot_cost()


# In[ ]:




