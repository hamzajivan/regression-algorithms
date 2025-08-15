#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[4]:

import numpy as np
import matplotlib.pyplot as plt

class MultipleRegModel:
    def __init__(self, x1_values,x2_values, y_values, alpha = 0.01, w1_init = 0, w2_init = 0, b_init = 0):
        self.x1 = np.array(x1_values)
        self.x2 = np.array(x2_values)
        self.y = np.array(y_values)
        self.alpha = alpha
        self.w1 = w1_init
        self.w2 = w2_init
        self.b = b_init
        self.m = len(self.y)
        self.cost_history = []

    def predict(self):
        return self.w1*self.x1+ self.w2*self.x2 + self.b

    def cost(self):
        errors = self.predict() - self.y
        cost_prediction = np.sum(np.square(errors))/(2*self.m)
        return cost_prediction

    def step_gradient_descent(self):
        errors = self.predict() - self.y
        
        gradient_w1  = (np.sum(errors*self.x1))/self.m
        self.w1 = self.w1 - self.alpha*(gradient_w1)
        
        gradient_w2  = (np.sum(errors*self.x2))/self.m
        self.w2 = self.w2 - self.alpha*(gradient_w2)
        
        gradient_b  = (np.sum(errors))/self.m
        self.b = self.b - self.alpha*(gradient_b)
    
    def train(self, iterations =10):
        for i in range(iterations):
            self.step_gradient_descent()
            w1 = round(self.w1, 5)
            w2 = round(self.w2, 5)
            b = round(self.b, 5)
            cost_val = round (self.cost(), 5)
            self.cost_history.append(cost_val) 
        print(f"Iter 1000: w1={self.w1:.5f}, w2={self.w2:.5f}, b={self.b:.5f}, cost_val={self.cost():.5f}")
        print(f"Multiple Linear Regression Equation: {self.w1:.5f}x1+{self.w2:.5f}x2+{self.b:.5f}")
        
    def plot_cost(self):
        plt.plot(range(1, len(self.cost_history) + 1), self.cost_history)
        plt.xlabel("Iteration")
        plt.ylabel("Cost J(w1, w2, b)")
        plt.title("Cost function over iterations")
        plt.show()
            
def get_values(name):
    return list(map(float, input(f"Enter {name} values separated by commas: ").split(",")))

x1_values = get_values("x1")
x2_values = get_values("x2")
y_values = get_values("y")
alpha  = float(input("Enter the learning rate:"))
iterations = int(input("Enter number of training iterations: "))

model = MultipleRegModel(
    x1_values=x1_values,
    x2_values=x2_values,
    y_values=y_values,
    w1_init=0,
    w2_init=0,
    b_init=0,
    alpha=alpha
)

model.train(iterations)

model.plot_cost()


# In[ ]:




