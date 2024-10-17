# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:39:44 2024

@author: Asus
"""

import numpy as np
import sympy as sym
class Optimization:
    
    def __init__(self,fx_input,x0,alpha,n):
        self.x0 = x0
        self.alpha = alpha
        self.n_iter = n
        self.fx_input = fx_input
        

    def Compute_fx(self,fx_input):

        
        return fx_input

    def Compute_df_dx(self):
        fx = self.Compute_fx(self.fx_input)
        df_dx = sym.diff(fx,'x')
        return df_dx
        
    def Compute_gradient_descent(self):    
        
        x_old = self.x0     
        for i in range(self.n_iter):  
            df_dx = self.Compute_df_dx()
            df_dx_eval = df_dx.evalf(subs={'x': x_old})
            x_new = x_old - self.alpha*df_dx_eval
            x_old = x_new
        return x_new

#User input    
x = sym.Symbol('x')
#fx_input = (x*x*sym.cos(x)-x)/10.    
fx_input = x*x - 5*x  

    
test = Optimization(fx_input,6,0.2,50)

x = test.Compute_gradient_descent()
print(x)