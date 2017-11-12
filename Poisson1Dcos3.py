#!/opt/local/bin/python3

"""
    Solve the 1D perturbed Poisson's equation
    using Neural Network with activation function cos(x)
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

# define solution space phi(t)
def phi(t,a,b,c,d):
    s = 0.0
    for i in range(n):
        s += d[i] * np.cos( a * i * t + b[i] )
    return s + c

# compute the energy error
def H1err(a,b,c,d):
    def dphi(t,a,b,c,d):
        s = 0.0
        for i in range(n):
            s -= d[i] * a * i * np.sin( a * i * t + b[i] )
        return s
    def graddiff2(t):
        return (gradu(t)-dphi(t,a,b,c,d))**2
    npt   = 511
    tquad = np.arange(-1,1,2.0/npt)
    yquad = graddiff2(tquad)
    return sqrt( integrate.trapz(yquad, tquad) )

# compute the L2 error
def L2err(a,b,c,d):
    def diff2(t):
        return (u(t)-phi(t,a,b,c,d))**2
    npt   = 511
    tquad = np.arange(-1,1,2.0/npt)
    yquad = diff2(tquad)
    return sqrt( integrate.trapz(yquad, tquad) )

# define energy function: Part 1, (grad u, grad u) / 2
def F1(a,b,c,d):
    s = 0.0
    for i in range(n):
        for j in range(n):
            ad = a * i * a * j * d[i] * d[j]
            if abs(a*i) < 1E-12 and abs(a*j) < 1E-12:
                s += ad * sin(b[i]) * sin(b[j])
            elif i == j:
                s += ad * 0.25 * (+ sin(a*j - b[i])*sin(a*j - b[j])       \
                                  + sin(a*j + b[i])*sin(a*j + b[j])       \
                                  + cos(a*j - b[i])*cos(a*j - b[j])       \
                                  + cos(a*j + b[i])*cos(a*j + b[j])       \
                                  - sin(a*j - b[j])*cos(a*j - b[i])/(a*j) \
                                  - sin(a*j + b[j])*cos(a*j + b[i])/(a*j) )
            else:
                s += ad * 0.5 * (- i*sin(a*j - b[j])*cos(a*i - b[i])  \
                                 - i*sin(a*j + b[j])*cos(a*i + b[i])  \
                                 + j*sin(a*i - b[i])*cos(a*j - b[j])  \
                                 + j*sin(a*i + b[i])*cos(a*j + b[j])) \
                              / (a*i**2 - a*j**2)
    return s

# define energy function: Part 2, (u, u) / 2
def F2(a,b,c,d):
    s = c**2
    for i in range(n):
        for j in range(n):
            ad = d[i] * d[j]
            if abs(a*i) < 1E-12 and abs(a*j) < 1E-12:
                s += ad * cos(b[i]) * cos(b[j])
            elif i == j:
                s += ad * 0.25 * (+ sin(a*j - b[i])*sin(a*j - b[j])       \
                                  + sin(a*j + b[i])*sin(a*j + b[j])       \
                                  + cos(a*j - b[i])*cos(a*j - b[j])       \
                                  + cos(a*j + b[i])*cos(a*j + b[j])       \
                                  + sin(a*j - b[i])*cos(a*j - b[j])/(a*j) \
                                  + sin(a*j + b[i])*cos(a*j + b[j])/(a*j) )
            else:
                s += ad * 0.5 * (+ i*sin(a*i - b[i])*cos(a*j - b[j])  \
                                 + i*sin(a*i + b[i])*cos(a*j + b[j])  \
                                 - j*sin(a*j - b[j])*cos(a*i - b[i])  \
                                 - j*sin(a*j + b[j])*cos(a*i + b[i])) \
                              / (a*i**2 - a*j**2)
        if abs(a*i) < 1E-12:
            s += 2 * c * d[i] * cos(b[i])
        else:
            s += c * d[i] * ( sin(a*i + b[i]) + sin(a*i - b[i]) ) / (a*i)
    return s

# define energy function: Part 3
def F3(a,b,c,d):
    # define the integrant for right-hand-side f*phi
    def f_phi(t):
        return phi(t,a,b,c,d) * f(t)
    # res = integrate.quad(f_phi, -1, 1, limit=100)
    nquad = 511
    tquad = np.linspace(-1,1,nquad) # generate a uniform mesh
    yquad = f_phi(tquad)
    res   = integrate.trapz(yquad, tquad)
    return res

# define energy function, x = [a,b,d]:
def F(x):
    a = x[0]
    b = x[1:1+n]
    c = x[1+n]
    d = x[2+n:2+2*n]
    return F1(a,b,c,d) + F2(a,b,c,d) - F3(a,b,c,d)

# define Jacobian of energy function:
def JacF(x):
    a = x[0]
    b = x[1:1+n]
    c = x[1+n]
    d = x[2+n:2+2*n]
    jac = np.zeros(len(x))

    nquad  = 129
    tquad  = np.linspace(-1,1,nquad) # generate a uniform mesh
    yquad1 = np.zeros(nquad)
    yquad2 = np.zeros(nquad)

    # prepare some sums on quadrature points
    s1 = np.zeros(nquad)
    s2 = np.zeros(nquad)
    s3 = np.zeros(nquad)
    for j in range(n):
        s1 -= d[j] * np.sin(a*j*tquad+b[j]) * j
        s2 += d[j] * np.cos(a*j*tquad+b[j])
        s3 -= d[j] * np.cos(a*j*tquad+b[j]) * j**2

    # compute partial derivatives for a, c
    yquad1 = s1 * a * ( s1 + s3 * a * tquad ) + ( s2 + c - f(tquad) ) * s1 * tquad
    yquad2 = s2 + c - f(tquad)
    jac[0  ] = integrate.trapz(yquad1, tquad)
    jac[1+n] = integrate.trapz(yquad2, tquad)

    # compute partial derivatives for b, d
    for i in range(n):
        yquad1 = - s1 * a**2 * i * d[i] * np.cos(a*i*tquad+b[i]) \
                 - ( s2 + c )    * d[i] * np.sin(a*i*tquad+b[i]) \
                 + f(tquad)      * d[i] * np.sin(a*i*tquad+b[i])
        yquad2 = - s1 * a**2 * i        * np.sin(a*i*tquad+b[i]) \
                 + ( s2 + c )           * np.cos(a*i*tquad+b[i]) \
                 - f(tquad)             * np.cos(a*i*tquad+b[i])
        jac[  1+i] = integrate.trapz(yquad1, tquad)
        jac[2+n+i] = integrate.trapz(yquad2, tquad)

    return jac

#===== main program begins from here =====
time_start = time()

K      = 10
n      = 6
#== Level  1     2     3     4     5      6  ==
maxits = [200,  300,  400,  500,  600,   1500 ]
fctols = [1E-3, 1E-5, 1E-7, 1E-9, 1E-11, 1E-13]
numlvl = 3
USE_GLOBAL_MIN = True # False
numhop = 10

# give an initial guess
np.random.seed(234567891)
x0 = (np.random.random(2*n+2)-0.5)*2

print( 'K =', K, ', n =', n*2**(numlvl-1), ', cos3')
print( 'Exact energy:', exact_energy() )
print( 'Initial energy:', F(x0) )
print( '--------------------------------------------------------------' )

format_min = "Success = %d, F(u) = %10.5e, Niter = %d, Nfeval = %d"
format_err = "n = %3d, L2err = %12.7e, H1err = %12.7e"

# call a multi-level minimization process
for l in range(numlvl-1):
    # call a minimizer to find a global min
    opt = {'ftol':fctols[l], 'maxiter':maxits[l]}
    res = minimize(F, x0, method='SLSQP', jac=JacF, options=opt)
    sol = res.x
    a   = sol[0]
    b   = sol[1:1+n]
    c   = sol[1+n]
    d   = sol[2+n:2+2*n]

    # output solution state and error
    print( format_min % (res.success, res.fun, res.nit, res.nfev)  )
    print( format_err % (n, L2err(a,b,c,d), H1err(a,b,c,d)) )
    print( '--------------------------------------------------------------' )

    # form a finer level initial guess
    n2 = n*2
    x0 = np.zeros(2*n2+2)
    x0[0]           = sol[0]                      # a
    x0[1:1+n]       = sol[1:1+n]                  # b old
    x0[1+n:n2+1]    = (np.random.random(n)-0.5)*2 # b new
    x0[n2+1]        = sol[1+n]                    # c
    x0[n2+2:n2+2+n] = sol[2+n:2+n2]               # d old
    x0[n2+2+n:]     = (np.random.random(n)-0.5)*2 # d new
    n = n2

# call a minimizer to find a global min for the finest level
opt = {'ftol':fctols[numlvl-1], 'maxiter':maxits[numlvl-1]}
res = minimize(F, x0, method='SLSQP', jac=JacF, options=opt)
sol = res.x
a   = sol[0]
b   = sol[1:1+n]
c   = sol[1+n]
d   = sol[2+n:2+2*n]

# output final solution state and error
print( format_min % (res.success, res.fun, res.nit, res.nfev)  )
print( format_err % (n, L2err(a,b,c,d), H1err(a,b,c,d)) )
print( '--------------------------------------------------------------' )

# tested SA in msa2
# v0 =  np.ones(len(x0))
# [sol,minres] = MomentSA2(F,JacF,0.1,100.0,0.1,x0,v0,100,5,[-1.0,1.0])

# call a global minimizer to improve quality
if USE_GLOBAL_MIN:
    x0 = sol
    minkwargs = {"method":"L-BFGS-B", "jac":JacF}
    res = basinhopping(F, x0, niter=numhop, T=0.2, minimizer_kwargs=minkwargs)
    sol = res.x
    a   = sol[0]
    b   = sol[1:1+n]
    c   = sol[1+n]
    d   = sol[2+n:2+2*n]

    # output solution state and error after global optimization
    format_min = "Objective F(u) = %10.5e, Niter = %d, Nfeval = %d"
    print( res.message[0] )
    print( format_min % (res.fun, res.nit, res.nfev)  )
    print( format_err % (n, L2err(a,b,c,d), H1err(a,b,c,d)) )
    print( '--------------------------------------------------------------' )

print( 'CPU time used:', time()-time_start, 'seconds' )

# output final energy and plot the solution
npt = 256
tpt = np.arange(-1,1,2.0/npt)
ypt = phi(tpt,a,b,c,d)
upt = u(tpt) # exact solution
plt.plot(tpt,upt,'r:')
plt.plot(tpt,ypt,'k')
plt.title( 'n = ' + str(n) + ', final energy = ' + str(F(sol)) )
#plt.show()
#np.save('data_abd.npy', sol)
