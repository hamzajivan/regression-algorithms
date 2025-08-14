#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

class LinearRegModel:
    def __init__(self, x_values, y_values, alpha = 0.01, w_init = 0, b_init = 0):
        self.x = np.array(x_values)
        self.y = np.array(y_values)
        self.alpha = alpha
        self.w = w_init
        self.b = b_init
        self.m = len(self.y)
        self.cost_history = []

    def predict(self):
        return self.w * self.x + self.b

    def cost(self):
        errors = self.predict() - self.y
        cost_prediction = np.sum(np.square(errors))/(2*self.m)
        return cost_prediction

    def step_gradient_descent(self):
        errors = self.predict() - self.y
        gradient_w  = (np.sum(errors*self.x))/self.m
        self.w = self.w - self.alpha*(gradient_w)
        gradient_b  = (np.sum(errors))/self.m
        self.b = self.b - self.alpha*(gradient_b)
    
    def train(self, iterations =10):
        for i in range(iterations):
            self.step_gradient_descent()
            w = round(self.w, 5)
            b = round(self.b, 5)
            cost_val = round (self.cost(), 5)
            self.cost_history.append(cost_val) 
        print(f"Iter 1000: w={self.w:.5f}, b={self.b:.5f}, cost_val={self.cost():.5f}")
        print(f"Linear Regression Equation: {self.w:.5f}x+{self.b:.5f}")
        
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

model = LinearRegModel(
    x_values=x_values,
    y_values=y_values,
    w_init=0,
    b_init=0,
    alpha=alpha
)

model.train(iterations)

model.plot_cost()

