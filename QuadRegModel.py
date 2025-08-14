#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot as plt

class QuadRegModel:
    def __init__(self, x_vals, y_vals, alpha, a, b, c):
        self.x = np.array(x_vals)
        self.y = np.array(y_vals)
        self.alpha = alpha
        self.a = a
        self.b = b
        self.c = c
        self.m = len(y_vals)
        self.cost_history = []

    def predict(self):
        return (self.a*np.square(self.x)) + (self.b*self.x) + self.c

    def cost_fxn(self):
        error = np.sum(np.square(self.predict() - self.y))
        return error/(2*self.m)
        
    def step_gradient(self):
        error = self.predict() - self.y
        derivative_a = np.sum(error*np.square(self.x))/self.m
        derivative_b = (np.sum(error*self.x))/self.m
        derivative_c = (np.sum(error))/self.m
        self.a = self.a - self.alpha * derivative_a
        self.b = self.b - self.alpha * derivative_b
        self.c = self.c - self.alpha * derivative_c

    def train(self, iterations=100):
        for i in range(iterations):
            self.step_gradient()
            a = np.round(self.a, 5)
            b = np.round(self.b, 5)
            c = np.round(self.c, 5)
            cost_val  = round(self.cost_fxn(), 5)
            self.cost_history.append(cost_val)  
        print(f"Iter {i+1}: a= {self.a: 5f}, b= {self.b: 5f}, c= {self.c:5f}, cost_val= {self.cost_fxn(): 5f}")
        print(f"Quadratic Function of best fit: {self.a: 5f}x^2+{self.b: 5f}x+{self.c:5f}")
        
    def plot_cost(self):
        plt.plot(range(1, len(self.cost_history) + 1), self.cost_history)
        plt.xlabel("Iteration")
        plt.ylabel("Cost J(w, b)")
        plt.title("Cost function over iterations")
        plt.show()
        
def get_values(name):
    return list(map(float, input(f"Enter {name} values separated by commas:").split(",")))


x = get_values("x")
y = get_values("y")
alpha = float(input(f"Enter the learning rate:"))
iterations = int(input(f"Enter the number of iterations:"))

model = QuadRegModel(
    x_vals = x,
    y_vals = y, 
    a = 0,
    b = 0, 
    c = 0, 
    alpha = alpha)

model.train(iterations)
model.plot_cost()


# In[ ]:




