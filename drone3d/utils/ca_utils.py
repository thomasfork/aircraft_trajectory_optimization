''' general purpose casadi untilities'''
import casadi as ca
import numpy as np

def ca_pos_abs(x, eps = 1e-3):
    '''
    smooth, positive apporoximation to abs(x)
    meant for tire slip ratios, where result must be nonzero
    '''
    return ca.sqrt(x**2 + eps**2)

def ca_abs(x):
    '''
    absolute value in casadi
    do not use for tire slip ratios
    used for quadratic drag: c * v * abs(v)
    '''
    return ca.if_else(x > 0, x, -x)

def ca_sign(x, eps = 1e-3):
    ''' smooth apporoximation to sign(x)'''
    return x / ca_pos_abs(x, eps)

def unpack_solution_helper(label, arg, out):
    '''
    utility for unpacking a subset of (arg)
    into a casadi array and then a numpy array

    primarily used for unpacking the
    long variable vector of an NLP into a
    specific array, ie. a 4 x N array of vehicle states
    '''
    f = ca.Function(label, [arg], out)
    return lambda arg: np.array(f(arg)).squeeze()
