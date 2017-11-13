#!/opt/local/bin/python3

"""
    Solve the 1D perturbed Poisson's equation
    using Neural Network with activation function ReLU(x)
    stochastic gradient descent
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

# compute the gradient of solution
def gradu(t):
    return 4*np.arctan(K*t)*(t**3-t) + (1-t**2)**2*K/(1+(K*t)**2)

# compute the right-hand-side
def f(t):
    return u(t) + 2 * K**3*t * (1-t**2)**2 / ((K*t)**2+1)**2 \
                + 8 * K*t    * (1-t**2)    / ((K*t)**2+1) \
                - np.arctan(K*t) * (12*t**2-4)

# compute the exact min energy
# exact solution u(x) = arctan(K*x) * (1-x**2)**2
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

# define ReLU3 function and its derivative
def ReLU(x):
    return x**3 * (x > 0) #return x * (x > 0)

def dReLU(x):
    return 3*x**2 * (x > 0) #return 1 * (x > 0)

# define solution space phi(t)
def phi(t,a,b,d):
    s = 0.0
    for i in range(n):
        s += d[i] * ReLU( a[i] * t + b[i] )
    return s

# compute the energy error
def H1err(a,b,d):
    def dphi(t,a,b,d):
        s = 0.0
        for i in range(n):
            s += d[i] * a[i] * dReLU( a[i] * t + b[i] )
        return s
    def graddiff2(t):
        return (gradu(t)-dphi(t,a,b,d))**2
    npt   = 511
    tquad = np.arange(-1,1,2.0/npt)
    yquad = graddiff2(tquad)
    return sqrt( integrate.trapz(yquad, tquad) )

# compute the L2 error
def L2err(a,b,d):
    def diff2(t):
        return (u(t)-phi(t,a,b,d))**2
    npt   = 511
    tquad = np.arange(-1,1,2.0/npt)
    yquad = diff2(tquad)
    return sqrt( integrate.trapz(yquad, tquad) )

# define energy function: Part 1, (grad u, grad u) / 2
def F1(a,b,d):
    def gphi_gphi(t):
        return dReLU(a[i] * t + b[i]) * dReLU(a[j] * t + b[j])
    s = 0.0
    nquad = 511
    tquad = np.linspace(-1,1,nquad) # generate a uniform mesh
    for i in range(n):
        for j in range(n):
            yquad = gphi_gphi(tquad)
            res   = integrate.trapz(yquad, tquad)
            s    += a[i] * a[j] * d[i] * d[j] * res
    return 0.5 * s

# define energy function: Part 2, (u, u) / 2
def F2(a,b,d):
    def phi_phi(t):
        return ReLU(a[i] * t + b[i]) * ReLU(a[j] * t + b[j])
    s = 0.0
    nquad = 511
    tquad = np.linspace(-1,1,nquad) # generate a uniform mesh
    for i in range(n):
        for j in range(n):
            yquad = phi_phi(tquad)
            res   = integrate.trapz(yquad, tquad)
            s    += d[i] * d[j] * res
    return 0.5 * s

# define energy function: Part 3, (f, u)
def F3(a,b,d):
    # define the integrant for right-hand-side f*phi
    def f_phi(t):
        return phi(t,a,b,d) * f(t)
    # compute res = integrate.quad(f_phi, -1, 1, limit=100)
    nquad = 511
    tquad = np.linspace(-1,1,nquad) # generate a uniform mesh
    yquad = f_phi(tquad)
    res   = integrate.trapz(yquad, tquad)
    return res

# define energy function:
def F(x):
    a = x[0:n]
    b = x[n:2*n]
    d = x[2*n:3*n]
    return F1(a,b,d) + F2(a,b,d) - F3(a,b,d)

# define Jacobian of energy function:
def JacF(x):
    def d2ReLU(x):
        return 6*x * (x > 0)
    a = x[0:n]
    b = x[n:2*n]
    d = x[2*n:3*n]
    jac = np.zeros(len(x))

    nquad  = 129
    tquad  = np.linspace(-1,1,nquad) # generate a uniform mesh
    yquad1 = np.zeros(nquad)
    yquad2 = np.zeros(nquad)
    yquad3 = np.zeros(nquad)

    # prepare some sums on quadrature points
    s1     = np.zeros(nquad)
    s2     = np.zeros(nquad)
    for j in range(n):
        s1 += a[j] * d[j] * dReLU(a[j]*tquad+b[j])
        s2 +=        d[j] *  ReLU(a[j]*tquad+b[j])

    # compute partial derivatives for a, c and d
    for i in range(n):
        yquad1 =   s1 * (         d[i] *  dReLU(a[i]*tquad+b[i]) \
                         + a[i] * d[i] * d2ReLU(a[i]*tquad+b[i]) * tquad ) \
                 + s2           * d[i] *  dReLU(a[i]*tquad+b[i]) * tquad \
                 - f(tquad)     * d[i] *  dReLU(a[i]*tquad+b[i]) * tquad
        yquad2 =   s1 *    a[i] * d[i] * d2ReLU(a[i]*tquad+b[i]) \
                 + s2           * d[i] *  dReLU(a[i]*tquad+b[i]) \
                 - f(tquad)     * d[i] *  dReLU(a[i]*tquad+b[i])
        yquad3 =   s1 *    a[i]        *  dReLU(a[i]*tquad+b[i]) \
                 + s2                  *   ReLU(a[i]*tquad+b[i]) \
                 - f(tquad)            *   ReLU(a[i]*tquad+b[i])
        jac[    i] = integrate.trapz(yquad1, tquad)
        jac[  n+i] = integrate.trapz(yquad2, tquad)
        jac[2*n+i] = integrate.trapz(yquad3, tquad)

    return jac

# define stochastic Jacobian of energy function:
def JacF_rand(x):
    def d2ReLU(x):
        return 6*x * (x > 0)
    a = x[0:n]
    b = x[n:2*n]
    d = x[2*n:3*n]
    jac = np.zeros(len(x))

    alpha  = 0.001
    nquad  = 100
    tquad  = (np.random.random(nquad)-0.5)*2
    yquad1 = np.zeros(nquad)
    yquad2 = np.zeros(nquad)
    yquad3 = np.zeros(nquad)

    # prepare some sums on quadrature points
    s1     = np.zeros(nquad)
    s2     = np.zeros(nquad)
    for j in range(n):
        s1 += a[j] * d[j] * dReLU(a[j]*tquad+b[j])
        s2 +=        d[j] *  ReLU(a[j]*tquad+b[j])

    # compute partial derivatives for a, c and d
    for i in range(n):
        yquad1 =   s1 * (         d[i] *  dReLU(a[i]*tquad+b[i]) \
                         + a[i] * d[i] * d2ReLU(a[i]*tquad+b[i]) * tquad ) \
                 + s2           * d[i] *  dReLU(a[i]*tquad+b[i]) * tquad \
                 - f(tquad)     * d[i] *  dReLU(a[i]*tquad+b[i]) * tquad
        yquad2 =   s1 *    a[i] * d[i] * d2ReLU(a[i]*tquad+b[i]) \
                 + s2           * d[i] *  dReLU(a[i]*tquad+b[i]) \
                 - f(tquad)     * d[i] *  dReLU(a[i]*tquad+b[i])
        yquad3 =   s1 *    a[i]        *  dReLU(a[i]*tquad+b[i]) \
                 + s2                  *   ReLU(a[i]*tquad+b[i]) \
                 - f(tquad)            *   ReLU(a[i]*tquad+b[i])
        jac[    i] = np.sum(yquad1)/nquad
        jac[  n+i] = np.sum(yquad2)/nquad
        jac[2*n+i] = np.sum(yquad3)/nquad

    return jac * alpha

# give a simple gradient descent method
def GradientDescent(x, numIterations):
    for i in range(numIterations):
        x -= JacF_rand(x)
    return x

#===== main program begins from here =====
time_start = time()

K      = 10
n      = 6
#== Level  1     2     3     4     5      6  ==
maxits = [200,  300,  400,  500,  600,   1500 ]
fctols = [1E-3, 1E-5, 1E-7, 1E-9, 1E-11, 1E-13]
numlvl = 4
nit    = 10000
numhop = 100
USE_GLOBAL_MIN = True # False

# give an initial guess
np.random.seed(234567891)
x0 = (np.random.random(3*n)-0.5)*2

print( 'K =', K, ', n =', n*2**(numlvl-1), ', relu3, SGD')
print( 'Exact energy:', exact_energy() )
print( 'Initial energy:', F(x0) )
print( '--------------------------------------------------------------' )

format_min = "Objective F(u) = %10.5e, Niter = %d, Nfeval = %d"
format_err = "n = %3d, L2err = %12.7e, H1err = %12.7e"

# call a multi-level minimization process
for l in range(numlvl-1):
    # call a minimizer to find a local min
    opt = {'ftol':fctols[l], 'maxiter':maxits[l]}
    sol = GradientDescent(x0, nit)
    a   = sol[0:n]
    b   = sol[n:2*n]
    d   = sol[2*n:3*n]

    # output solution state and error
    print( format_min % (F(sol), nit, nit) )
    print( format_err % (n, L2err(a,b,d), H1err(a,b,d)) )
    print( '--------------------------------------------------------------' )

    # form a finer level initial guess
    n2 = n*2
    x0 = np.zeros(3*n2)
    x0[0:n]         = sol[0:n]                    # a old
    x0[n:n2]        = (np.random.random(n)-0.5)*2 # a new
    x0[n2:n2+n]     = sol[n:2*n]                  # b old
    x0[n2+n:2*n2]   = (np.random.random(n)-0.5)*2 # b new
    x0[n2*2:n2*2+n] = sol[2*n:3*n]                # d old
    #x0[n2*2+n:]     = (np.random.random(n)-0.5)*2 # d new
    n = n2

# call a minimizer to find a local min on the finest level
opt = {'ftol':fctols[numlvl-1], 'maxiter':maxits[numlvl-1]}
sol = GradientDescent(x0, nit)
a   = sol[0:n]
b   = sol[n:2*n]
d   = sol[2*n:3*n]

# output final solution state and error
print( format_min % (F(sol), nit, nit) )
print( format_err % (n, L2err(a,b,d), H1err(a,b,d)) )
print( '--------------------------------------------------------------' )

# call a global minimizer to improve quality
if USE_GLOBAL_MIN:
    x0 = sol
    minkwargs = {"method":"L-BFGS-B", "jac":JacF}
    res = basinhopping(F, x0, niter=numhop, T=10, minimizer_kwargs=minkwargs)
    sol = res.x
    a   = sol[0:n]
    b   = sol[n:2*n]
    d   = sol[2*n:3*n]

    # output solution state and error after global optimization
    print( res.message[0] )
    print( format_min % (res.fun, res.nit, res.nfev)  )
    print( format_err % (n, L2err(a,b,d), H1err(a,b,d)) )
    print( '--------------------------------------------------------------' )

print( 'CPU time used:', time()-time_start, 'seconds' )

# output final energy and plot the solution
npt = 256
tpt = np.arange(-1,1,2.0/npt)
ypt = phi(tpt,a,b,d)
upt = u(tpt) # exact solution
plt.plot(tpt,upt,'r:')
plt.plot(tpt,ypt,'k')
plt.title( 'n = ' + str(n) + ', final energy = ' + str(F(sol)) )
plt.show()
#np.save('data_abd.npy', sol)
