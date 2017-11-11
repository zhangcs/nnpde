#! /usr/bin/python2.7

"""
Different Descent Algorithms

vkp1 = (1-mu*dt)*vk-dt/gamma*grad(f(xk))+sigma/gamma*dW
xkp1 = xk+dt*vkp1

"""

#from __future__ import print_function
import warnings
import numpy as np
import numpy.random as rd
import os
import sys
from scipy.optimize import *
from scipy.optimize.linesearch import (line_search_wolfe1, line_search_wolfe2,
                         line_search_wolfe2 as line_search,
                         LineSearchWarning)

def MomentSA2(Func,Jac,mu,gamma,amp,x_init,v_init,itmax,K,bounds=None,DirDoF=None):
    print("Moment 2 Simulated Annealing...")
    N = len(x_init)
    sol_x = np.zeros(N,'d')
    min_x = np.copy(x_init)
    min_energy = Func(x_init)
    B=0.1
    C=0.1

    DoF = range(N)
    if DirDoF!=None:
        for i in DirDoF:
            DoF.remove(i)

    print("Initial Gradient = ", np.linalg.norm(Jac(x_init)), "Initial Energy = ", min_energy)
    for j in range(K):
        it = 2
        res = Func(x_init)
        x = np.copy(x_init)
        v = np.copy(v_init)
        z = np.ones(N,'d')
        while ( (it < itmax) ):
            dt = 1.0/np.log(float(it))**0.5
            sigma = amp*np.sqrt(gamma/(np.log(it)*dt))*1.0/float(it)
            z = (1.0 -B*dt)*z + C*v*dt + sigma*rd.normal(0.0,1.0,N)*np.sqrt(dt)
            b = Jac(x)
            v = (1.0-mu*dt)*v - dt*b/gamma - z*dt
            new_x = x + v*dt

            if (bounds!=None and len(bounds)==2 ):
                for i in DoF:
                    if (new_x[i]>bounds[0] and new_x[i]<bounds[1]):
                        x[i]=new_x[i]
            else:
                x = np.copy(new_x)

            it+=1

        res = np.linalg.norm(Jac(x))
        Energy = Func(x)
        print("j = ",j," Gradient = ",res, "Energy = ", Energy)
        sol_x += x/float(K)
        if Energy < min_energy:
            min_energy = Energy
            min_x = np.copy(x)

    Energy = Func(sol_x)
    if Energy < min_energy:
        min_energy = Energy
        min_x = np.copy(sol_x)

    return [min_x,min_energy]

def LBFGS_MSA2(Func,Jac,mu,gamma,amp,x_init,v_init,itmax,K,bounds=None,DirDoF=None):
    print("LBFGS Moment 2 Simulated Annealing...")
    N = len(x_init)
    sol_x = np.zeros(N,'d')
    min_x = np.copy(x_init)
    min_energy = Func(x_init)
    B=0.1
    C=0.1

    DoF = range(N)
    if DirDoF!=None:
        for i in DirDoF:
            DoF.remove(i)

    print("Initial Gradient = ", np.linalg.norm(Jac(x_init)), "Initial Energy = ", min_energy)
    m = 10

    for j in range(K):
        q = np.zeros(N,'d')
        s = np.zeros((m,N),'d')
        y = np.zeros((m,N),'d')
        alpha = np.zeros(m,'d')
        z = np.ones(N,'d')

        Hk = np.eye(N,N)
        y[0,:] = Jac(x_init)
        s[0,:] = Jac(x_init)
        it = 2
        x = np.copy(x_init)
        v = np.copy(v_init)

        # Line Search
        old_fval = Func(x)
        gfk = Jac(x)
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        # STEP 1 (it=2)
        dt = 1.0/np.log(float(it))**0.5
        sigma = amp*np.sqrt(gamma/(np.log(it)*dt))*1.0/float(it)

        b = Jac(x)
        z = (1.0-B*dt)*z + C*v*dt + sigma*rd.normal(0.0,1.0,N)*np.sqrt(dt)
        v = (1.0-mu*dt)*v - dt*b/gamma - z*dt
        new_x = x + v*dt

        old_x = np.copy(x)
        if (bounds!=None and len(bounds)==2 ):
            for i in DoF:
                if (new_x[i]>bounds[0] and new_x[i]<bounds[1]):
                    x[i]=new_x[i]
        else:
            x = np.copy(new_x)

        y[0,:] = Jac(x)-b
        s[0,:] = x-old_x
        it+=1

        while ( (it < itmax) ):
            dt = 1.0/np.log(float(it))
            sigma = amp*np.sqrt(gamma/(np.log(it)*dt))*1.0/float(it)
            z = (1.0-B*dt)*z + C*v*dt + sigma*rd.normal(0.0,1.0,N)*np.sqrt(dt)


            ## CALL THE DESCENT (THIS WORKS)
            n_corrs = max(min(it-m-2, m),1)
            hess_inv = LbfgsInvHessProduct(s[:n_corrs], y[:n_corrs])
            b = hess_inv(Jac(x))

            # No LINE SEARCH
            v = (1.0-mu*dt)*v - dt*b/gamma - z*dt
            new_x = x + v*dt
            old_x = np.copy(x)

            if (bounds!=None and len(bounds)==2 ):
                for i in DoF:
                    if (new_x[i]>bounds[0] and new_x[i]<bounds[1]):
                        x[i]=new_x[i]
            else:
                x = np.copy(new_x)

            for i in range(0,max(min(m-1,it-m-2),1)):
                y[i+1,:] = y[i,:]
                s[i+1,:] = s[i,:]
            y[0,:] = Jac(x)-Jac(old_x)
            s[0,:] = x-old_x

            it+=1

        res = np.linalg.norm(Jac(x))
        Energy = Func(x)
        print("j = ",j," Gradient = ",res, "Energy = ", Energy)
        sol_x += x/float(K)
        if Energy < min_energy:
            min_energy = Energy
            min_x = np.copy(x)

    Energy = Func(sol_x)
    if Energy < min_energy:
        min_energy = Energy
        min_x = np.copy(sol_x)

    return [min_x,min_energy]
