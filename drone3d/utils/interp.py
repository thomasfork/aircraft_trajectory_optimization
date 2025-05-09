''' radial basis function interpolation'''

import time
import numpy as np
import casadi as ca
import scipy.interpolate

def linear_interpolant(x_data, y_data) -> ca.Function:
    '''
    univariate linear interpolation,
    out-of bounds data is clipped to last value
    '''
    x_interp = ca.SX.sym('x')
    y_lin = ca.pw_lin(x_interp, [*x_data, x_data[-1] + 1], [*y_data, y_data[-1]])
    return ca.Function('y', [x_interp], [y_lin])

def pchip_interpolant(x_data, y_data, extrapolate = 'const') -> ca.Function:
    '''
    univariate PCHIP interpolation
    close to linear interpolation but more derivatives are defined
    useful for approximating linear interpolation with more derivatives present
    '''
    spline = scipy.interpolate.PchipInterpolator(x_data, y_data)
    return scipy_spline_to_casadi(spline, extrapolate)

def spline_interpolant(x_data, y_data,
                bc_type = 'periodic',
                extrapolate = 'const',
                fast = False) -> ca.Function:
    '''
    univariate cubic spline interpolation
    for full documentation see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    main useful boundary conditions are:
    periodic
    not-a-knot
    clamped

    can optionally return a fast evaluation spline that is MX only

    It is possible to package a scipy spline as a casadi function
    however this does not result in any
    speed up because the function will still be evaluated 1 by 1
    by casadi.

    This returns a spline that can be used in CasADi's framework, but for
    fastest runtime one should use the scipy spline in a vectorized manner
    or other spline libraries.
    '''
    spline = scipy.interpolate.CubicSpline(x_data, y_data, bc_type = bc_type)
    if fast:
        return _fast_ca_spline(spline)
    return scipy_spline_to_casadi(spline, extrapolate)

def scipy_spline_to_casadi(spline, extrapolate = 'const'):
    '''
    converts a scipy spline function to a casadi function

    by default, truncates to the last spline value, however it can extrapolate linearly as well,
    which is useful when fitting a spline to a centerline (ie. spline_surface.py)
    '''
    k_c = spline.c
    k_x = spline.x
    k_f = spline(k_x[-1])

    x_fit = ca.SX.sym('s')
    x_0 = ca.pw_const(x_fit, k_x, [k_x[0], *k_x])
    c_0 = ca.pw_const(x_fit, k_x, [k_c[3,0], *k_c[3,:], k_f])
    if extrapolate == 'const':
        c_1 = ca.pw_const(x_fit, k_x, [0,        *k_c[2,:], 0])
    elif extrapolate == 'linear':
        c_1 = ca.pw_const(x_fit, k_x, [k_c[2,0], *k_c[2,:], spline(k_x[-1],1)])
    else:
        raise NotImplementedError(f'Unrecognized extrapolation key: {extrapolate}')


    c_2 = ca.pw_const(x_fit, k_x, [0,        *k_c[1,:], 0])
    c_3 = ca.pw_const(x_fit, k_x, [0,        *k_c[0,:], 0])

    x_rel = x_fit - x_0
    y_fit = c_0 + c_1*x_rel + c_2*x_rel**2 + c_3*x_rel**3
    f_y = ca.Function('k', [x_fit], [y_fit])

    return f_y

def _fast_ca_spline(spline):
    '''
    a spline that evaluates faster for large numbers of knots
    but is MX only
    particularly useful for long racetracks
    with problems that fix path length during optimization

    this is slightly slower at runtime than ca.interpolant('', 'bspline',...)
    but allows the added flexibility of scipy's spline fitting.

    Due to the use of ca.low this should not be differentiated,
    the first three derivatives are returned to avoid the need for this
    '''
    k_c = spline.c
    k_x = spline.x

    s = ca.MX.sym('s')
    sc = ca.if_else(s < k_x.min(), k_x.min(), ca.if_else(s > k_x.max(), k_x.max(), s))
    idx = ca.low(k_x,s)


    k_x = ca.MX(ca.DM(k_x))
    k_c = ca.MX(ca.DM(k_c))

    offset = k_x[idx]
    coeffs = k_c[:,idx]

    s_rel = sc - offset

    # fits for spline and its derivatives
    s0 = s_rel**ca.DM(np.arange(3,-1,-1))
    s1 = s_rel**ca.DM(np.arange(2,-1,-1)) * ca.DM([3,2,1])
    s2 = s_rel**ca.DM(np.arange(1,-1,-1)) * ca.DM([6,2])
    s3 = ca.DM([6])

    fit0 = ca.dot(coeffs, s0)
    fit1 = ca.dot(coeffs[:-1], s1)
    fit2 = ca.dot(coeffs[:-2], s2)
    fit3 = ca.dot(coeffs[:-3], s3)

    f = ca.Function('x',[s],[fit0])
    f1 = ca.Function('x',[s],[fit1])
    f2 = ca.Function('x',[s],[fit2])
    f3 = ca.Function('x',[s],[fit3])

    return f, f1, f2, f3

def demo_spline():
    ''' demo of spline fit timing with many points '''
    s = np.linspace(0,2*np.pi,1000)
    x = np.sin(6*s)
    y = np.sin(7*s)
    x[-1] = x[0]
    y[-1] = y[0]

    N = 10000
    s_eval = np.linspace(0, 2*np.pi, N)

    fx = spline_interpolant(s, x)

    t0 = time.time()
    fx(s_eval)
    print(time.time() - t0)

    fx = spline_interpolant(s, x, fast = True)[0]

    t0 = time.time()
    fx(s_eval)
    print(time.time() - t0)

    fx = ca.interpolant('x','bspline', [s], x)

    t0 = time.time()
    fx(s_eval)
    print(time.time() - t0)

if __name__ == '__main__':
    demo_spline()
