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

# define NEW solution space phi(t)
def phi(t,a,b,c,d):
    s = 0
    for i in range(n):
        s += np.cos( a[i] * t + b[i] )
    return s * d / n + c

# compute the energy error
def H1err(a,b,c,d):
    def dphi(t,a,b,c,d):
        s = 0
        for i in range(n):
            s -= a[i] * np.sin( a[i] * t + b[i] )
        return s * d / n
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
            ad = a[i] * a[j]
            if abs(a[i]) < 1E-12 and abs(a[j]) < 1E-12:
                s += ad * sin(b[i]) * sin(b[j])
            elif abs( a[i] + a[j] ) < 1E-12:
                s += ad * 0.25 * (- sin(a[j] - b[i])*sin(a[j] + b[j])      \
                                  - sin(a[j] + b[i])*sin(a[j] - b[j])      \
                                  - cos(a[j] - b[i])*cos(a[j] + b[j])      \
                                  - cos(a[j] + b[i])*cos(a[j] - b[j])      \
                                  + sin(a[j] - b[i])*cos(a[j] + b[j])/a[j] \
                                  + sin(a[j] + b[i])*cos(a[j] - b[j])/a[j] )
            elif abs( a[i] - a[j] ) < 1E-12:
                s += ad * 0.25 * (+ sin(a[j] - b[i])*sin(a[j] - b[j])      \
                                  + sin(a[j] + b[i])*sin(a[j] + b[j])      \
                                  + cos(a[j] - b[i])*cos(a[j] - b[j])      \
                                  + cos(a[j] + b[i])*cos(a[j] + b[j])      \
                                  - sin(a[j] - b[i])*cos(a[j] - b[j])/a[j] \
                                  - sin(a[j] + b[i])*cos(a[j] + b[j])/a[j] )
            else:
                s += ad * 0.5 * (- a[i]*sin(a[j] - b[j])*cos(a[i] - b[i])  \
                                 - a[i]*sin(a[j] + b[j])*cos(a[i] + b[i])  \
                                 + a[j]*sin(a[i] - b[i])*cos(a[j] - b[j])  \
                                 + a[j]*sin(a[i] + b[i])*cos(a[j] + b[j])) \
                              / (a[i]**2 - a[j]**2)
    return s * d**2 / n**2

# define energy function: Part 2, (u, u) / 2
def F2(a,b,c,d):
    s  = c**2
    dn = d**2/n**2
    cd = c*d/n
    for i in range(n):
        for j in range(n):
            if abs(a[i]) < 1E-12 and abs(a[j]) < 1E-12:
                s += dn * cos(b[i]) * cos(b[j])
            elif abs( a[i] + a[j] ) < 1E-12:
                s += dn * 0.25 * (+ sin(a[j] - b[i])*sin(a[j] + b[j])      \
                                  + sin(a[j] + b[i])*sin(a[j] - b[j])      \
                                  + cos(a[j] - b[i])*cos(a[j] + b[j])      \
                                  + cos(a[j] + b[i])*cos(a[j] - b[j])      \
                                  + sin(a[j] - b[i])*cos(a[j] + b[j])/a[j] \
                                  + sin(a[j] + b[i])*cos(a[j] - b[j])/a[j] )
            elif abs( a[i] - a[j] ) < 1E-12:
                s += dn * 0.25 * (+ sin(a[j] - b[i])*sin(a[j] - b[j])      \
                                  + sin(a[j] + b[i])*sin(a[j] + b[j])      \
                                  + cos(a[j] - b[i])*cos(a[j] - b[j])      \
                                  + cos(a[j] + b[i])*cos(a[j] + b[j])      \
                                  + sin(a[j] - b[i])*cos(a[j] - b[j])/a[j] \
                                  + sin(a[j] + b[i])*cos(a[j] + b[j])/a[j] )
            else:
                s += dn * 0.50 * (+ a[i]*sin(a[i] - b[i])*cos(a[j] - b[j])  \
                                  + a[i]*sin(a[i] + b[i])*cos(a[j] + b[j])  \
                                  - a[j]*sin(a[j] - b[j])*cos(a[i] - b[i])  \
                                  - a[j]*sin(a[j] + b[j])*cos(a[i] + b[i])) \
                                  / (a[i]**2 - a[j]**2)
        if abs(a[i]) < 1E-12:
            s += cd * 2 * cos(b[i])
        else:
            s += cd * ( sin(a[i] + b[i]) + sin(a[i] - b[i]) ) / a[i]
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

# define energy function, x = [a,b,c,d]:
def F(x):
    a = x[0:n]
    b = x[n:2*n]
    c = x[2*n]
    d = x[2*n+1]
    return F1(a,b,c,d) + F2(a,b,c,d) - F3(a,b,c,d)

# define Jacobian of energy function:
def JacF(x):
    a = x[0:n]
    b = x[n:2*n]
    c = x[2*n]
    d = x[2*n+1]
    dn = d/n
    jac = np.zeros(len(x))

    nquad  = 129
    tquad  = np.linspace(-1,1,nquad) # generate a uniform mesh
    yquad1 = np.zeros(nquad)
    yquad2 = np.zeros(nquad)

    # prepare some sums on quadrature points
    s1     = np.zeros(nquad)
    s2     = np.zeros(nquad)
    for j in range(n):
        s1 -= a[j] * np.sin(a[j]*tquad+b[j])
        s2 +=        np.cos(a[j]*tquad+b[j])

    # compute partial derivatives for a and b
    for i in range(n):
        yquad1 = - s1  * dn * (        dn * np.sin(a[i]*tquad+b[i]) \
                              + a[i] * dn * np.cos(a[i]*tquad+b[i]) * tquad ) \
                 - (s2 * dn + c)     * dn * np.sin(a[i]*tquad+b[i]) * tquad \
                 + f(tquad)          * dn * np.sin(a[i]*tquad+b[i]) * tquad
        yquad2 = - s1 * dn    * a[i] * dn * np.cos(a[i]*tquad+b[i]) \
                 - (s2 * dn + c)     * dn * np.sin(a[i]*tquad+b[i]) \
                 + f(tquad)          * dn * np.sin(a[i]*tquad+b[i])
        jac[  i] = integrate.trapz(yquad1, tquad)
        jac[n+i] = integrate.trapz(yquad2, tquad)

    # compute partial derivatives for c and d
    yquad1     = s2 * dn + c - f(tquad)
    jac[2*n  ] = integrate.trapz(yquad1, tquad)
    yquad2     = s1 * dn * s1 / n + (s2 * dn + c - f(tquad)) * s2 / n
    jac[2*n+1] = integrate.trapz(yquad2, tquad)

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

print( 'K =', K, ', n =', n*2**(numlvl-1), ', cos2')
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
    a   = sol[0:n]
    b   = sol[n:2*n]
    c   = sol[2*n]
    d   = sol[2*n+1]
    
    # output solution state and error
    print( format_min % (res.success, res.fun, res.nit, res.nfev)  )
    print( format_err % (n, L2err(a,b,c,d), H1err(a,b,c,d)) )
    print( '--------------------------------------------------------------' )

    # form a finer level initial guess
    n2 = n*2
    x0 = np.zeros(2*n2+2)
    x0[0:n]         = sol[0:n]                    # a old
    x0[n:n2]        = (np.random.random(n)-0.5)*2 # a new
    x0[n2:n2+n]     = sol[n:2*n]                  # b old
    x0[n2+n:2*n2]   = (np.random.random(n)-0.5)*2 # b new
    x0[2*n2]        = sol[2*n]                    # c
    x0[2*n2+1]      = sol[2*n+1]                  # d
    n = n2

# call a minimizer to find a global min for the finest level
opt = {'ftol':fctols[numlvl-1], 'maxiter':maxits[numlvl-1]}
res = minimize(F, x0, method='SLSQP', jac=JacF, options=opt)
sol = res.x
a   = sol[0:n]
b   = sol[n:2*n]
c   = sol[2*n]
d   = sol[2*n+1]

# output final solution state and error
print( format_min % (res.success, res.fun, res.nit, res.nfev)  )
print( format_err % (n, L2err(a,b,c,d), H1err(a,b,c,d)) )
print( '--------------------------------------------------------------' )

# call a global minimizer to improve quality
if USE_GLOBAL_MIN:
    x0 = sol
    minkwargs = {"method":"L-BFGS-B", "jac":JacF}
    res = basinhopping(F, x0, niter=numhop, T=0.2, minimizer_kwargs=minkwargs)
    sol = res.x
    a   = sol[0:n]
    b   = sol[n:2*n]
    c   = sol[2*n]
    d   = sol[2*n+1]

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
