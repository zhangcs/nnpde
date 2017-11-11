#!/opt/local/bin/python3

"""
    Solve the 2D Poisson's equation on L-shaped domain
    using Neural Network with activation function cos(x)
"""

from time import time
from math import *
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

# compute the exact solution
def u(t):
    return 0.0

# compute the gradient of solution
def gradu(t):
    return np.zeros(len(t))

# compute the right-hand-side
def f(t1,t2):
    return 1.0

# compute the exact min energy
def exact_energy():
    def gradu2(t):
        return gradu(t)**2
    def u2(t):
        return u(t)**2
    def fu(t):
        return f(t) * u(t)
    res1 = integrate.quad(gradu2, -1, 1, limit=100)
    res2 = integrate.quad(u2, -1, 1, limit=100)
    res3 = integrate.quad(fu, -1, 1, limit=100)
    return 0.5 * (res1[0]+res2[0]) - res3[0]

# define solution space phi(t)
def phi(t1,t2,a,b,d):
    s = 0.0
    for i in range(n):
        s += d[i] * np.cos( a[i,0]*t1 + a[i,1]*t2 + b[i] )
    return s

# define energy function: Part 1, (grad u, grad u) / 2
def F1(a,b,d):
    def dphi2(t1,t2):
        return sin(a[i,0]*t1 + a[i,1]*t2 + b[i]) * sin(a[j,0]*t1 + a[j,1]*t2 + b[j])
    s = 0.0
    print('evaluate F1')
    opt = {'epsrel':1e-06, 'limit':100}
    for i in range(n):
        for j in range(n):
            print(i, j, a[i,0], a[i,1], a[j,0], a[j,1])
            ad  = np.dot(a[i,:],a[j,:]) * d[i] * d[j]
            res = integrate.nquad(dphi2, [[-1,1],[-1,1]], opts=[opt,opt])
            s  += ad * res[0]
    return 0.5 * s

# define energy function: Part 2, (u, u) / 2
def F2(a,b,d):
    s = 0.0
    for i in range(n):
        for j in range(n):
            ad = d[i] * d[j]
            #TBA
    return s

# define energy function: Part 3
def F3(a,b,d):
    # define the integrant for right-hand-side f*phi
    def f_phi(t1,t2):
        return f(t1,t2) * phi(t1,t2,a,b,d)
    print('evaluate F3')
    opt = {'epsrel':1e-06, 'limit':100}
    res = integrate.nquad(f_phi, [[-1, 1],[-1,1]], opts=[opt,opt])
    return res[0]

# define energy function, x = [a,b,d]:
def F(x):
    a = x[0:2*n].reshape(n,2)
    b = x[2*n:3*n]
    d = x[3*n:4*n]
    return F1(a,b,d) + F2(a,b,d) - F3(a,b,d)

# define Jacobian of energy function:
def JacF(x):
    jac = np.zeros(len(x))
    return jac

#===== main program begins from here =====
time_start = time()

n = 4

# give an initial guess
x0 = np.random.random(4*n)
print( 'initial energy:', F(x0) )

# call a multi-level minimization process
maxits = [100,  200,  300]
fctols = [1E-3, 1E-4, 1E-6]

for l in range(2):
    # call a minimizer to find a global min
    opt = {'ftol':fctols[l], 'maxiter':maxits[l], 'disp': True}
    res = minimize(F, x0, method='SLSQP', options=opt)
    # form a finer level initial guess
    n2 = n*2
    x0 = np.random.random(4*n2)
    x0[0:n*2]   = res.x[0:n*2]
    x0[n*4:n*5] = res.x[n*2:n*3]
    x0[n*6:n*7] = res.x[3*n:4*n]
    x0[n*7:n*8] = np.zeros(n)
    n = n2

# call a minimizer to find a global min for the finest level
opt = {'ftol':1E-8, 'maxiter':500, 'disp': True}
res = minimize(F, x0, method='SLSQP', options=opt)
#res = minimize(F, x0, method='SLSQP', jac=JacF, options=opt)
sol = res.x

time_end = time()
print( 'CPU time used:', time_end-time_start, 'seconds' )
np.save('data_abd.npy', sol)

# output final energy and plot the solution
npt = 100
tpt1 = np.arange(-1,1,2.0/npt)
tpt2 = np.arange(-1,1,2.0/npt)
a = sol[0:2*n].reshape(n,2)
b = sol[2*n:3*n]
c = sol[3*n:4*n]
ypt = phi(tpt1, tpt2, a, b, d)
Axes3D.plot_surface(tpt1,tp2,ypt,'r:')
plt.title( 'Solution with n = ' + str(n) )
plt.show()
