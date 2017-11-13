#!/opt/local/bin/python3

"""
    Interpolation using ReLU as a basis function
"""

from time import time
from math import *
from scipy.optimize import minimize, basinhopping
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

# compute the exact solution
def u(t):
    return np.arctan(K*t) * (1-t**2)**2

# define ReLU function and alpha function
def ReLU(x):
    return x * (x > 0)

def alpha(x):
    return ReLU(x) - ReLU(x-1)

# define solution space phi(t)
def phi(x, t, u, h):
    n = len(t)
    s = u[0] * ( ReLU( -(x-t[1])/h ) - ReLU( -(x-t[1])/h-1 ) )
    for i in range(1,n-1):
        s += u[i] * ( ReLU( (x-t[i])/h+1) - 2* ReLU( (x-t[i])/h ) + ReLU( (x-t[i])/h-1 ))
    s += u[n-1] * ( ReLU( (x-t[n-1])/h+1 ) - ReLU( (x-t[n-1])/h ) )
    return s

# output final energy and plot the solution
K = 10
n = 9
h = 2.0/(n-1)
t = np.linspace(-1,1,n)
y = u(t) # exact solution

npt = 129
tpt = np.linspace(-1,1,npt)
upt = u(tpt)
ypt = phi(tpt, t, y, h)

plt.plot(tpt,upt,'r:')
plt.plot(tpt,ypt,'k-')
plt.show()
