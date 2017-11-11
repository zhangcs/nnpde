#!/opt/local/bin/python3

"""
    Compute simbolic integrals in energy functional
"""

from sympy import *

x  = symbols('x')
a  = symbols('a')
i  = symbols('i')
j  = symbols('j')
bi = symbols('bi')
bj = symbols('bj')
K  = symbols('K')

##################################################
# Compute (Gaussian)''
##################################################

print( diff(exp(-x**2/(2*K**2))/K, x) )

print( diff(diff(exp(-x**2/(2*K**2))/K, x), x) )

print( '==============================' )

##################################################
# Compute \int_{-1,1} \grad u(x) \grad u(x) dx
##################################################

F1 = integrate(sin(a*i*x+bi) * sin(a*j*x+bj), (x,-1,1))
print(F1)
pprint(pretty(F1))

print( '==============================' )

##################################################
# Compute \int_{-1,1} u(x) u(x) dx
##################################################

F2 = integrate(cos(a*i*x+bi) * cos(a*j*x+bj), (x,-1,1))
print(F2)
pprint(pretty(F2))

print( '==============================' )

##################################################
# Compute \int_{-1,1} 1 u(x) dx
##################################################

F3 = integrate(cos(a*i*x+bi), (x,-1,1))
print(F3)
pprint(pretty(F3))
