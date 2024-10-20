# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:39:44 2024

@author: Asus
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy.utilities.lambdify import lambdify


class Optimization:
    
    def __init__(self,fx_input,alpha,n,*x):
        
        self.x0 = x[0]
        self.y0 =  x[1]
        self.alpha = alpha
        self.n_iter = n
        self.fx_input = fx_input

    def Plot_function(self,fx_input):

        x,y = sym.symbols('x,y')

        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)

        # Lambdfy and vectorize symbolic function
        fxy = np.vectorize(lambdify([x, y], fx_input))
        z = fxy(X,Y)

        # Plot the surface.
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)    
        plt.show()
        

    def Compute_fx(self,fx_input):
        return fx_input

    def Compute_df_dx(self):

        fx = self.Compute_fx(self.fx_input)
        df_dx = sym.diff(fx,'x')
        df_dy = sym.diff(fx,'y')

        return df_dx, df_dy
        
    def Compute_gradient_descent(self):  
        x = sym.Symbol('x')
        y = sym.Symbol('y')
        x_old = self.x0 
        y_old = self.y0    
        df_dx, df_dy = self.Compute_df_dx()

        df_dx_numeric = lambdify((x, y), df_dx)
        df_dy_numeric = lambdify((x, y), df_dy)


        for i in range(self.n_iter):  
            df_dx_eval =  df_dx_numeric(x_old, y_old)
            df_dy_eval =  df_dy_numeric(x_old, y_old)

            #update of vectors
            x_new = x_old - self.alpha*df_dx_eval
            y_new = y_old - self.alpha*df_dy_eval
            x_old = x_new
            y_old = y_new

        return x_new, y_new
        


#User input    
x = sym.Symbol('x')
y = sym.Symbol('y')
a = 1
b =100
fx_input = (a-x)* (a-x) + b*(y - x*x)*(y - x*x)
x = (0,0)   


test = Optimization(fx_input,0.002,2000,*x)
test.Plot_function(fx_input)
x = test.Compute_gradient_descent()
print('Minimum of fx located at: ',x)